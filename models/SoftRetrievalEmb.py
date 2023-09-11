import copy
import json
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from utils import ANSWER_DATA_PATH
import pytorch_model_summary as pms


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth')


class RetrievalSelection(nn.Module):

    def __init__(self, hidden_state_size, top_k, train_skip_self):
        super(RetrievalSelection, self).__init__()
        self.top_k = top_k
        self.train_skip_self = train_skip_self
        self.attn_qst_query = nn.MultiheadAttention(hidden_state_size, 4, dropout=0.1)

    def soft_attention_select_2d(self, query_feat, kv_feat, mask):
        kv_feat = kv_feat.permute(1, 0, 2)
        query_feat = query_feat.unsqueeze(0)
        _, temp_weights = self.attn_qst_query(query_feat, kv_feat, kv_feat,
                                              attn_mask=mask, key_padding_mask=None)
        return temp_weights

    def find_top_k_fast(self, temp_weights, modality):
        _, top_k_index_sort = torch.topk(temp_weights, k=min(self.top_k, temp_weights.size()[-1]), dim=-1)
        modality_new = modality[torch.arange(modality.size(0)).unsqueeze(1), top_k_index_sort.squeeze(1)]
        return modality_new, top_k_index_sort

    def select_top_k_fast(self, patch_feat, top_k_index_sort):
        patch_select_new = patch_feat[torch.arange(patch_feat.size(0)).unsqueeze(1), top_k_index_sort.squeeze(1)]
        return patch_select_new

    def forward(self, query, key, value=None):
        if self.train_skip_self and self.training:
            mask = None
        else:
            mask = None
        temp_weights = self.soft_attention_select_2d(key, query, mask)
        top_k_query, top_k_index_sort = self.find_top_k_fast(temp_weights, query)
        if value is None:
            return temp_weights, top_k_query, None
        else:
            top_k_value = self.select_top_k_fast(value, top_k_index_sort)
        return temp_weights, top_k_query, top_k_value


class TrainableSoftTopK(nn.Module):
    def __init__(self, hidden_state_size=512, top_k=25, train_skip_self=False):
        super().__init__()
        self.temp_vid = nn.Parameter(data=torch.zeros(1, device='cuda'), requires_grad=True)
        self.temp_aud = nn.Parameter(data=torch.zeros(1, device='cuda'), requires_grad=True)
        self.temp_ocr = nn.Parameter(data=torch.zeros(1, device='cuda'), requires_grad=True)
        self.retrieval_v = RetrievalSelection(hidden_state_size=hidden_state_size, top_k=top_k,
                                              train_skip_self=train_skip_self)
        self.retrieval_a = RetrievalSelection(hidden_state_size=hidden_state_size, top_k=top_k,
                                              train_skip_self=train_skip_self)
        self.retrieval_o = RetrievalSelection(hidden_state_size=hidden_state_size, top_k=top_k,
                                              train_skip_self=train_skip_self)

    def forward(self, q1, k1, q2, k2, q3=None, k3=None, mask=None, **args):
        vid_gate = torch.nn.functional.sigmoid(self.temp_vid)
        aud_gate = torch.nn.functional.sigmoid(self.temp_aud)
        ocr_gate = torch.nn.functional.sigmoid(self.temp_ocr)

        q1 = torch.nn.functional.normalize(q1, p=2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, p=2, dim=-1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, p=2, dim=-1)
        ### Special Case for OCR ###
        if not (q3 is None):
            q3 = torch.nn.functional.normalize(q3, p=2, dim=-1)
            k3 = torch.nn.functional.normalize(k3, p=2, dim=-1)
            pass

        # scores1 = torch.nn.functional.cosine_similarity(q1.unsqueeze(1), k1, dim=-1)
        scores1, _, _ = self.retrieval_v(k1, q1)

        if mask is not None:
            scores1 *= mask
        scores1 = torch.clip(scores1, 0, 1)

        # scores2 = torch.nn.functional.cosine_similarity(q2.unsqueeze(1), k2, dim=-1)
        scores2, _, _ = self.retrieval_a(k2, q2)
        if mask is not None:
            scores2 *= mask
        scores2 = torch.clip(scores2, 0, 1)

        ### Special Case for OCR ###
        if not (q3 is None):
            # scores3 = torch.nn.functional.cosine_similarity(q3.unsqueeze(1), k3, dim=-1)
            scores3, _, _ = self.retrieval_o(k3, q3)

            if mask is not None:
                scores3 *= mask
            scores3 = torch.clip(scores3, 0, 1)
            ocr_part = scores3 * ocr_gate
        else:
            scores3 = 0
            attention3 = 0
            ocr_part = 0

        vid_part = scores1 * vid_gate
        aud_part = scores2 * aud_gate
        attention = vid_part + aud_part + ocr_part
        B, L, C = attention.size()
        attention = attention.contiguous().view(B, C, L)
        return attention


class MultiRetrievalAugmentedEmbeddingV5(nn.Module):
    def __init__(self,
                 v_dim=768,
                 l_dim=384,
                 o_dim=None,
                 ocr_dim=768,
                 use_embedding=False,
                 common_dropout=0.25,
                 use_aux_loss=0,
                 overconfidence_loss=0.0,
                 top_k=25,
                 train_skip_self=False):
        super().__init__()
        self.overconfidence_loss = overconfidence_loss
        self.use_embedding = use_embedding
        self.v_dim = v_dim
        self.q_dim = l_dim
        self.ocr_dim = ocr_dim
        self.common_dropout = common_dropout
        self.use_aux_loss = use_aux_loss
        if o_dim is None:
            self.o_dim = self.q_dim
        else:
            self.o_dim = o_dim
        self.emb_dim = 768
        # Common Dropout #
        self.common_dropout = nn.Dropout1d(p=self.common_dropout)
        # Common BLOCKS #
        self.common_vision_base = nn.Identity()
        self.v_block = nn.Sequential(self.common_vision_base)
        self.v_block2 = nn.Sequential(self.common_vision_base, self.common_dropout)

        self.common_audio_base = nn.Identity()
        self.a_block = nn.Sequential(self.common_audio_base)
        self.a_block2 = nn.Sequential(self.common_audio_base, self.common_dropout)

        self.common_ocr_base = nn.Identity()
        self.ocr_block = nn.Sequential(self.common_ocr_base)
        self.ocr_block2 = nn.Sequential(self.common_ocr_base, self.common_dropout)

        if not self.use_embedding:
            self.common_option_base = nn.Identity()
            self.o_block = nn.Sequential(self.common_option_base)
            self.o_block2 = nn.Sequential(self.common_option_base, self.common_dropout)
        else:
            self.embedding_library = nn.Embedding(num_embeddings=6370, embedding_dim=self.emb_dim)

        self.similarity_suggestor = TrainableSoftTopK(hidden_state_size=self.emb_dim, top_k=top_k,
                                                      train_skip_self=train_skip_self)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_fn = MulticlassAccuracy(num_classes=3, average='weighted')

    def calc_loss(self, scores, gt):
        """
        Calculate Cross-Entropy Loss
        :param scores: The unnormalised logits
        :param gt: A list of the true labels
        :return:
        """
        if not isinstance(gt, torch.Tensor):
            gt = torch.LongTensor(gt).to('cuda')
        if len(gt.size()) > 1:
            gt = gt.squeeze(1)

        return self.loss_fn(scores, gt)

    def calc_acc(self, scores, gt):
        """
        Calculate Accuracy
        :param scores: The unnormalised logits
        :param gt: A list of the true labels
        :return:
        """

        if not isinstance(gt, torch.Tensor):
            gt = torch.LongTensor(gt).to('cuda')
        if len(gt.size()) > 1:
            gt = gt.squeeze(1)
        scores = scores.argmax(dim=-1)
        if len(scores.size()) > 1:
            scores = scores.squeeze(1)
        return self.metric_fn(scores, gt)

    def forward(self, v, n_feats, aud, n_auds, ocr, n_ocrs, o, n_answ, gt_answer_int=None, **args):
        v = self.v_block(v)
        v2 = self.v_block2(n_feats)
        aud = self.a_block(aud)
        aud2 = self.a_block2(n_auds)
        if not self.use_embedding:
            o = self.o_block(o)
            o2 = self.o_block2(n_answ)
        else:
            o = self.embedding_library(o)
            o2 = self.embedding_library(n_answ)
        #############################################
        ### Mini-Mask ###
        if ocr.sum() > 0:
            ocr = self.ocr_block(ocr)
            ocr2 = self.ocr_block2(n_ocrs)
        else:
            ocr = None
            ocr2 = None
        mix_informed_answer = self.similarity_suggestor(v, v2, aud, aud2, ocr, ocr2)
        options_informed_answer = mix_informed_answer * o2
        options_informed_answer = options_informed_answer.sum(1)
        #############################################
        score_0 = options_informed_answer * o[:, 0, :]
        score_0 = score_0.sum(1).unsqueeze(1)
        score_1 = options_informed_answer * o[:, 1, :]
        score_1 = score_1.sum(1).unsqueeze(1)
        score_2 = options_informed_answer * o[:, 2, :]
        score_2 = score_2.sum(1).unsqueeze(1)
        if self.use_aux_loss > 0 and self.training:
            # Hard-Mine N extra embeddings #
            false_hard_mine = torch.randint(low=0, high=6369, size=(o.size()[0], self.use_aux_loss), device='cuda')
            false_hard_embeddings = self.embedding_library(false_hard_mine)

            # Attach them to the option_informed_answer #
            fake_scores = options_informed_answer.repeat(1, self.use_aux_loss, 1) * false_hard_embeddings
            fake_scores = fake_scores.sum(2)
            scores = torch.cat([score_0, score_1, score_2, fake_scores], dim=1)
        else:
            scores = torch.cat([score_0, score_1, score_2], dim=1)
        if gt_answer_int is not None:
            loss = self.calc_loss(scores, gt_answer_int)
            metric = self.calc_acc(scores[:, :3], gt_answer_int)
            return scores, loss, metric
        return scores, None, None


class IterableLookUpV2(Dataset):
    def __init__(self,
                 active_db,
                 cached_db=None,
                 model_emb=None,
                 answer_data_json=None,
                 top_k=25, use_embedding=False, lfoh=True, qo_folders=[]):
        self.use_embedding = use_embedding
        self.answer_data_json = answer_data_json
        self.model_emb = model_emb
        self.top_k = top_k
        self.lfoh = lfoh
        if cached_db is None:
            cached_db = active_db
        self.qo_folders = qo_folders
        self.reorder_db(active_db, cached_db)

        return

    def custom_collate_fn(self, batch):
        KEYS = batch[0].keys()
        NEW_BATCH = {k: [] for k in KEYS}
        forbidden_key = 'none'
        for entry in batch:
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    pass
                elif isinstance(v, list):
                    if isinstance(v[0], int):
                        v = torch.LongTensor(v)
                    else:
                        v = torch.stack(v, dim=0)
                elif isinstance(v, int):
                    v = torch.LongTensor([v])
                elif isinstance(v, str):
                    forbidden_key = k
                    pass
                v = v.to('cuda') if not isinstance(v, str) else v
                NEW_BATCH[k].append(v)

        for k in KEYS:
            if k != forbidden_key:
                NEW_BATCH[k] = torch.stack(NEW_BATCH[k], dim=0)
        # Expand Options #
        if not self.use_embedding:
            if self.lfoh:
                NEW_BATCH['o'] = torch.nn.functional.one_hot(NEW_BATCH['o'], num_classes=6370).squeeze(1).to('cuda')
                NEW_BATCH['o'] = NEW_BATCH['o'].float()
                NEW_BATCH['n_answers'] = torch.nn.functional.one_hot(NEW_BATCH['n_answers'], num_classes=6370).squeeze(
                    1).to(
                    'cuda')
                NEW_BATCH['n_answers'] = NEW_BATCH['n_answers'].float()
            else:
                pass
        else:
            NEW_BATCH['o'] = NEW_BATCH['o'].long()
            NEW_BATCH['n_answers'] = NEW_BATCH['n_answers'].long()
        return NEW_BATCH

    def reorder_db(self, active_db, cached_db):
        idx = 0
        db = {}
        # For each question in the active dataset #
        for active_entry in tqdm.tqdm(active_db):
            # Get YOUR VIDEO #
            active_video_frames = active_entry['frames'].unsqueeze(0)
            # Get YOUR AUDIO #
            active_audio_frames = active_entry['audio'].unsqueeze(0)
            # Get YOUR OCR #
            active_ocr_frames = active_entry['ocr']
            # Get YOUR QUESTIONS #
            for active_question_items in active_entry['mc_question']:
                # Get YOUR OPTIONS #
                if self.answer_data_json is not None:
                    active_option_frames = active_question_items['options']
                    active_option_frames = torch.LongTensor(
                        [self.answer_data_json[f]['int_index'] for f in active_option_frames])
                elif self.answer_data_json is None and self.lfoh is False:
                    ### Find and load the active Options ###
                    # question_file = os.path.join(self.qo_folders[0][0],
                    #                              active_entry['metadata']['video_id'].split('_')[1]) + '_sent.npy'
                    answer_file = os.path.join(self.qo_folders[0][1],
                                               active_entry['metadata']['video_id'].split('_')[1]) + '_sent.npy'
                    #question_feat = torch.from_numpy(np.load(question_file)).float()
                    active_option_frames = torch.from_numpy(np.load(answer_file)).float()
                # Get YOUR GT ANSWER (if any) #
                active_gt_frames = active_question_items['answer_id']
                # Get the same question in the cached dataset #
                cached_question_items = cached_db[active_question_items['question']]

                # Get the Video and the Audio and OCR Frames from it #
                cached_video_frames = torch.stack(cached_question_items['video_emb'], dim=0)
                cached_audio_frames = torch.stack(cached_question_items['audio_emb'], dim=0)
                cached_ocr_frames = torch.stack(cached_question_items['ocr_emb'], dim=0).squeeze(1)

                # Get the Answers from it #
                if self.answer_data_json is not None:
                    cached_answer_frames = torch.LongTensor(cached_question_items['answer'])
                elif self.answer_data_json is None and self.lfoh is False:
                    cached_answer_frames = torch.stack(cached_question_items['answer'], dim=0)

                # Find top k video frames #
                top_k_video_frames = cached_video_frames

                # Find top k audio frames #
                top_k_audio_frames = cached_audio_frames

                # Find top k ocr frames #
                top_k_ocr_frames = cached_ocr_frames

                # Find top k answers #
                top_k_answers = cached_answer_frames.squeeze(0)
                top_k_answers = top_k_answers

                db.update({idx: {
                    'v': active_video_frames.squeeze(0),
                    'aud': active_audio_frames.squeeze(0),
                    'ocr': active_ocr_frames.squeeze(0),
                    'q': active_question_items['question'],
                    'o': active_option_frames,
                    'n_feats': top_k_video_frames,
                    'n_auds': top_k_audio_frames,
                    'n_ocrs': top_k_ocr_frames,
                    'n_answers': top_k_answers,
                    'gt_answer_int': active_gt_frames,
                }})
                idx += 1
        self.final_ds = db
        return

    def __len__(self):
        return len(self.final_ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.final_ds[idx]


class SR_MCVQA_EMB:
    """Multiple-Choice vQA Model using Video Audio and OCR Inputs, Frequency  and Q/A as embedding pairs"""

    def __init__(self,
                 active_ds,
                 cache_ds=None,
                 model_emb='all-MiniLM-L12-v2',
                 look_for_one_hot=True,
                 use_embedding=False,
                 common_dropout=0.25,
                 use_aux_loss=0, model_version=5, top_k=25, train_skip_self=False, *args):
        self.top_k = top_k
        self.lfoh = look_for_one_hot
        self.use_embedding = use_embedding
        self.embedding_transformer = None
        self.model_version = model_version
        if self.model_version == 5:
            self.model = MultiRetrievalAugmentedEmbeddingV5(use_embedding=use_embedding,
                                                            common_dropout=common_dropout,
                                                            use_aux_loss=use_aux_loss, top_k=top_k,
                                                            train_skip_self=train_skip_self)
        self.model.to(device='cuda')
        v = torch.zeros(1, 768).to('cuda')
        aud = torch.zeros(1, 768).to('cuda')
        ocr = torch.ones(1, 768).to('cuda')
        if self.use_embedding:
            o = torch.zeros(1, 3, dtype=torch.long).to('cuda')
            n_answ = torch.zeros(1, 25, dtype=torch.long).to('cuda')
        else:
            if self.lfoh:
                o = torch.zeros(1, 3, 6370).to('cuda')
                n_answ = torch.zeros(1, 25, 6370).to('cuda')
            else:
                o = torch.zeros(1, 3, 512).to('cuda')
                n_answ = torch.zeros(1, 25, 512).to('cuda')
        n_feats = torch.zeros(1, 25, 768).to('cuda')
        n_auds = torch.zeros(1, 25, 768).to('cuda')
        n_ocrs = torch.ones(1, 25, 768).to('cuda')
        pms.summary(self.model, v, n_feats, aud, n_auds, ocr, n_ocrs, o, n_answ, show_input=True, print_summary=True)
        self._active_ds = copy.deepcopy(active_ds) if active_ds is not None else None
        if cache_ds is not None:
            self._cache_ds = self.build_video_answer_db_2(cache_ds)
        else:
            self._cache_ds = self.build_video_answer_db_2(active_ds)
        self.qo_folders = self._active_ds.qo_folders
        self.freeze_retrieval_part()

    def build_video_answer_db_2(self, dataset) -> Dict[str, Any]:
        """Builds a Video / Audio Embedding - Answer database.

        Returns a dictionary where the key is the question string and the value
        is a list of strings, each string a correct answer to a previous
        instance of the question.

        Args:
            dataset (PerceptionDataset): A dataset generator that provides video embeddings, questions and answers

        Returns:
            Dict[str, Any]: Dictionary where the key is question string and
            the value is list of full correct answers each time that question
            was asked in the db_dict.
        """
        if self.lfoh:
            print('Loading Answer Dict...\n')
            with open(ANSWER_DATA_PATH, 'r') as fin:
                self.answer_data_json = json.load(fin)
        else:
            self.answer_data_json = None

        question_db = {}
        print("Building Video Audio Dataset...\n")
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                try:
                    question_db[question['question']]
                except KeyError:
                    question_db[question['question']] = {'video_emb': [], 'audio_emb': [], 'ocr_emb': [], 'answer': [],
                                                         'answer_int': [], 'options': []}

                if self.lfoh:
                    answer = self.answer_data_json[question['options'][question['answer_id']]]['int_index']
                else:
                    answer = entry['clip_o_frames'][question['answer_id'], :]

                answer_int = int(question['answer_id'])
                video_emb = entry['frames'].cpu()
                audio_emb = entry['audio'].cpu()
                ocr_emb = entry['ocr'].cpu()

                if self.lfoh:
                    options = [self.answer_data_json[f]['int_index'] for f in question['options']]
                else:
                    options = entry['clip_o_frames']
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['audio_emb'].append(audio_emb)
                question_db[question['question']]['ocr_emb'].append(ocr_emb)
                question_db[question['question']]['answer'].append(answer)
                question_db[question['question']]['answer_int'].append(answer_int)
                question_db[question['question']]['options'].append(options)

        return question_db

    def fit(self, lr=0.0005, bs=128, epochs=100, val_external_dataset=None):
        FIXED_BATCH_SIZE = 1
        self.model.train()
        dataset = IterableLookUpV2(active_db=self._active_ds,
                                   cached_db=self._cache_ds,
                                   model_emb=self.embedding_transformer,
                                   answer_data_json=self.answer_data_json, use_embedding=self.use_embedding,
                                   lfoh=self.lfoh, qo_folders=self.qo_folders)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=FIXED_BATCH_SIZE,
                                shuffle=True,
                                pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        optimizer.zero_grad()

        for epoch_idx in range(epochs):
            epoch_loss = 0
            epoch_metric = 0
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                real_index = (epoch_idx * len(dataloader)) + step_idx
                v, \
                    aud, \
                    ocr, \
                    o, \
                    n_feats, \
                    n_answ, \
                    n_auds, \
                    n_ocrs, \
                    gt_answer_int = batch['v'], \
                    batch['aud'], \
                    batch['ocr'], \
                    batch['o'], \
                    batch['n_feats'], \
                    batch['n_answers'], \
                    batch['n_auds'], \
                    batch['n_ocrs'], \
                    batch['gt_answer_int']
                _, loss, metric = self.model(v=v,
                                             aud=aud,
                                             ocr=ocr,
                                             o=o,
                                             n_ids=None,
                                             n_feats=n_feats,
                                             n_answ=n_answ,
                                             n_auds=n_auds,
                                             n_ocrs=n_ocrs,
                                             gt_answer_int=gt_answer_int)

                epoch_loss += loss.detach().item()
                epoch_metric += metric.detach().item()
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Loss': epoch_loss / (step_idx + 1),
                     'Accuracy': epoch_metric / (step_idx + 1)})
                loss = loss / bs
                loss.backward()
                if ((step_idx + 1) % bs == 0) or (step_idx + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

            # END OF EPOCH EVAL #
            # self.eval(val_external_dataset)

        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'SR_MCVQA_EMB_Model{self.model_version}')

        # Freeze the video_model #
        self.model.eval()
        return

    def eval(self, val_dataset=None):
        FIXED_BATCH_SIZE = 1
        self.model.eval()
        if val_dataset is None:
            active_db = self._active_ds
        else:
            active_db = val_dataset
        dataset = IterableLookUpV2(active_db=active_db,
                                   cached_db=self._cache_ds,
                                   model_emb=self.embedding_transformer,
                                   answer_data_json=self.answer_data_json, use_embedding=self.use_embedding,
                                   lfoh=self.lfoh)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=FIXED_BATCH_SIZE,
                                shuffle=False,
                                pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)
        gg = []
        gt = []
        with torch.no_grad():
            print("---------------[VALIDATION] ---------------")
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                v, \
                    aud, \
                    ocr, \
                    o, \
                    n_feats, \
                    n_answ, \
                    n_auds, \
                    n_ocrs = batch['v'], \
                    batch['aud'], \
                    batch['ocr'], \
                    batch['o'], \
                    batch['n_feats'], \
                    batch['n_answers'], \
                    batch['n_auds'], \
                    batch['n_ocrs']
                _sim_scores, _, _ = self.model(v=v,
                                               aud=aud,
                                               ocr=ocr,
                                               o=o,
                                               n_ids=None,
                                               n_feats=n_feats,
                                               n_answ=n_answ,
                                               n_auds=n_auds,
                                               n_ocrs=n_ocrs,
                                               gt_answer_int=None)
                try:
                    for f in _sim_scores.squeeze().argmax(dim=-1):
                        gg.append(f)
                except:
                    for f in [_sim_scores.squeeze().argmax(dim=-1)]:
                        gg.append(f)
                try:
                    for g in batch['gt_answer_int'].squeeze():
                        gt.append(g)
                except:
                    for g in [batch['gt_answer_int'].squeeze()]:
                        gt.append(g)
        correct = 0
        total = 0
        for a, b in zip(gg, gt):
            if a == b:
                correct += 1
            total += 1
        print(f"[Validation Results]: Correct: {correct}  / Total: {total} / Acc: {correct / total}")
        return

    def answer_q(self,
                 frames,
                 question,
                 audio_frames=None,
                 ocr_frames=None,
                 option_frames=None,
                 n_ids=None, use_embedding=False,
                 **args):
        """
            Answer a multiple choice question.
            :param sample: Whether to sample or use argmax
            :param top_k: Filter the top-k similar results
            :param temp: Use temp when sampling
        """

        stored_response = self._cache_ds[question['question']]
        ### Rank answers based on incoming Frame ###

        if len(stored_response['video_emb'][0].size()) == 1:
            db_frames = torch.stack(stored_response['video_emb'], dim=0)
        elif len(stored_response['video_emb'][0].size()) == 2:
            db_frames = torch.cat(stored_response['video_emb'], dim=0)
        if audio_frames is not None:
            if len(stored_response['audio_emb'][0].size()) == 1:
                audb_frames = torch.stack(stored_response['audio_emb'], dim=0)
            elif len(stored_response['audio_emb'][0].size()) == 2:
                audb_frames = torch.cat(stored_response['audio_emb'], dim=0)
        if ocr_frames is not None:
            if len(stored_response['ocr_emb'][0].size()) == 1:
                ocrdb_frames = torch.stack(stored_response['ocr_emb'], dim=0)
            elif len(stored_response['ocr_emb'][0].size()) == 2:
                ocrdb_frames = torch.cat(stored_response['ocr_emb'], dim=0)
        if option_frames is not None:
            no_frames = stored_response['answer']

        _frames = frames.unsqueeze(0).to('cuda')
        _audio_frames = audio_frames.unsqueeze(0).to('cuda')
        _ocr_frames = ocr_frames.to('cuda')
        _db_frames = db_frames.unsqueeze(0).to('cuda')
        _audb_frames = audb_frames.unsqueeze(0).to('cuda')
        _ocrdb_frames = ocrdb_frames.unsqueeze(0).to('cuda')
        ### Fix incoming frames ###
        if not self.answer_data_json is None and option_frames is not None:
            o_ = torch.LongTensor([self.answer_data_json[f]['int_index'] for f in option_frames])
            if not self.use_embedding:
                o_ = torch.nn.functional.one_hot(o_, num_classes=6370).unsqueeze(0)
                o_ = o_.float()
            else:
                o_ = o_.unsqueeze(0)
            o_ = o_.to('cuda')
            ########
            if not self.use_embedding:
                _no_frames = torch.nn.functional.one_hot(torch.LongTensor([no_frames]), num_classes=6370)
                _no_frames = _no_frames.float()
            else:
                _no_frames = torch.LongTensor([no_frames])
            _no_frames = _no_frames.to('cuda')
            ### Fix incoming masks ###
            if n_ids is not None:
                _n_ids = None
        else:
            print("FIX THIIIIIIS")
        with torch.no_grad():
            _sim_scores, _, _ = self.model(
                v=_frames,
                aud=_audio_frames,
                ocr=_ocr_frames,
                o=o_,
                n_ids=None,
                n_feats=_db_frames,
                n_answ=_no_frames,
                n_auds=_audb_frames,
                n_ocrs=_ocrdb_frames,
                gt_answer_int=None)
        gg = _sim_scores.squeeze()
        answer_id = [gg.argmax().item()]
        answer = [stored_response['answer'][gg.argmax().item()]]
        return answer_id, answer

    def load_weights(self, path):
        data = torch.load(path)
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

    def freeze_retrieval_part(self):
        ### Freeze the retrieval mechanism ###

        return
