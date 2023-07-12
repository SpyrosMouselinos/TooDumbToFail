import copy
import json
import math
import os.path
import pickle
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from utils import ANSWER_DATA_PATH


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth')


class EncoderBlockV1(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=None, dropout=0.1, add_skip=False):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.add_skip = add_skip  # Whether to add skip connection at the layer
        # Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True,
                                               device='cuda')

        if dim_feedforward is None:
            dim_feedforward = 2 * embed_dim

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embed_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # Attention part
        attn_out, _ = self.self_attn(query=q, key=k, value=v)
        if self.add_skip:
            q = q + self.dropout(attn_out)
            q = self.norm1(q)
        else:
            q = attn_out

        # # MLP part
        # linear_out = self.linear_net(q)
        # q = q + self.dropout(linear_out)
        # q = self.norm2(q)

        return q


class MultiRetrievalAugmentedEmbeddingV1(nn.Module):
    def __init__(self, v_dim=768, l_dim=384, o_dim=None):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = l_dim
        if o_dim is None:
            self.o_dim = self.q_dim
        else:
            self.o_dim = o_dim
        self.emb_dim = self.v_dim
        # Common Dropout #
        self.common_dropout = nn.Dropout1d(p=0.0)
        # Common VISION BLOCKS #
        self.v_block = nn.Identity()
        self.q_block = nn.Identity()
        self.o_block = nn.Identity()
        self.transform_block = nn.Sequential(nn.Linear(self.v_dim + self.q_dim + self.o_dim, 128), nn.ReLU(),
                                             nn.Dropout(p=0.5), nn.Linear(128, 1))
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

    def forward(self, v, q, o, gt_answer_int=None, **args):
        v = self.v_block(v)
        q = self.q_block(q)
        o = self.o_block(o)
        combination1 = torch.cat([v, q, o[:, 0, :]], dim=1)
        combination2 = torch.cat([v, q, o[:, 1, :]], dim=1)
        combination3 = torch.cat([v, q, o[:, 2, :]], dim=1)
        score_0 = self.transform_block(combination1)
        score_1 = self.transform_block(combination2)
        score_2 = self.transform_block(combination3)
        scores = torch.cat([score_0, score_1, score_2], dim=1)
        if gt_answer_int is not None:
            loss = self.calc_loss(scores, gt_answer_int)
            metric = self.calc_acc(scores, gt_answer_int)
            return scores, loss, metric
        return scores, None, None


class EncoderBlockV2(nn.Module):
    def __init__(self, embed_dims, num_heads, dim_feedforwards=None, out_dims=None, dropout=0.1, add_skip=False):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.add_skip = add_skip  # Whether to add skip connection at the layer
        # Attention layer
        self.self_attn1 = nn.MultiheadAttention(embed_dim=embed_dims[0], num_heads=num_heads, dropout=dropout,
                                                batch_first=True,
                                                device='cuda')
        self.self_attn2 = nn.MultiheadAttention(embed_dim=embed_dims[1], num_heads=num_heads, dropout=dropout,
                                                batch_first=True,
                                                device='cuda')
        self.temp = nn.Parameter(data=0.5 * torch.ones(1, device='cuda'), requires_grad=True)

        if dim_feedforwards is None:
            dim_feedforwards = [2 * embed_dims[0], 2 * embed_dims[1]]

        if out_dims is None:
            out_dims = dim_feedforwards

        # Two-layer MLP
        self.linear_net1 = nn.Sequential(
            nn.Linear(embed_dims[0], dim_feedforwards[0]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforwards[0], out_dims[0]),
        )
        self.linear_net2 = nn.Sequential(
            nn.Linear(embed_dims[1], dim_feedforwards[1]),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforwards[1], out_dims[1]),
        )

        # Layers to apply in between the main layers
        self.norm11 = nn.LayerNorm(out_dims[0])
        self.norm12 = nn.LayerNorm(out_dims[1])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q1, k1, v1, q2, k2, v2, mask=None, **args):
        # Attention part
        a = torch.nn.functional.sigmoid(self.temp)
        attn_out1, _ = self.self_attn1(query=q1, key=k1, value=v1)
        attn_out2, _ = self.self_attn2(query=q2, key=k2, value=v2)
        if self.add_skip1:
            q1 = q1 + self.dropout(attn_out1)
            q1 = self.norm11(q1)
        else:
            q1 = attn_out1

        if self.add_skip2:
            q2 = q2 + self.dropout(attn_out2)
            q2 = self.norm12(q2)
        else:
            q2 = attn_out2

        # # MLP part
        # linear_out = self.linear_net(q)
        # q = q + self.dropout(linear_out)
        # q = self.norm2(q)
        out = q1 * a + (1 - a) * q2
        return out


class NoBrainEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(data=0.5 * torch.ones(1, device='cuda'), requires_grad=True)

    def forward(self, q1, k1, q2, k2, mask=None, **args):
        a = torch.nn.functional.sigmoid(self.temp)
        q1 = torch.nn.functional.normalize(q1, p=2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, p=2, dim=-1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, p=2, dim=-1)
        scores1 = torch.nn.functional.cosine_similarity(q1.unsqueeze(1), k1, dim=-1)
        if mask is not None:
            scores1 *= mask
        scores1 = torch.clip(scores1, 0, 1)
        scores2 = torch.nn.functional.cosine_similarity(q2.unsqueeze(1), k2, dim=-1)
        if mask is not None:
            scores2 *= mask
        scores2 = torch.clip(scores2, 0, 1)
        attention1 = F.softmax(scores1, dim=-1)
        attention2 = F.softmax(scores2, dim=-1)
        attention = attention1 * a + (1 - a) * attention2
        return attention


class TopKGroup(nn.Module):
    def __init__(self, top_k=25):
        super().__init__()
        self.top_k = top_k

    def forward(self, score_vector, **args):
        _, top_k_indices = torch.topk(score_vector, k=min(self.top_k, score_vector.size()[-1]), dim=-1)
        top_k_indices = top_k_indices.squeeze(0)
        mask = torch.zeros(size=(score_vector.size()[-1],), device='cuda')
        mask[top_k_indices] = 1
        mask = mask.unsqueeze(0)
        score_vector = score_vector * mask
        return score_vector


class MultiRetrievalAugmentedEmbeddingV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = NoBrainEncoderBlock()
        self.tkg = TopKGroup()
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

    def forward(self,
                v,
                aud,
                o=None,
                n_feats=None,
                n_ids=None,
                n_answ=None,
                n_auds=None,
                gt_answer_int=None, **args):

        if n_ids is not None:
            mask = 1.0 - (1.0 * (n_ids == -1))
            mask = mask.squeeze(1)
        else:
            mask = None
        #################################################################################
        aggr = self.bb(q1=v, k1=n_feats, q2=aud, k2=n_auds, mask=mask)
        aggr = self.tkg(aggr)
        #########################################
        if not o is None and not n_answ is None:
            aggr_ = aggr.unsqueeze(2)
            aggr_ = aggr_ * n_answ
            aggr_ = aggr_.sum(1)
            score_0 = aggr_ * o[:, 0, :]  # BS, E
            score_0 = score_0.sum(1).unsqueeze(1)
            score_1 = aggr_ * o[:, 1, :]  # BS, E
            score_1 = score_1.sum(1).unsqueeze(1)
            score_2 = aggr_ * o[:, 2, :]  # BS, E
            score_2 = score_2.sum(1).unsqueeze(1)
            scores = torch.cat([score_0, score_1, score_2], dim=1)
        else:
            scores = aggr

        if gt_answer_int is not None:
            loss = self.calc_loss(scores, gt_answer_int)
            metric = self.calc_acc(scores, gt_answer_int)
            return scores, loss, metric
        return scores, None, None


class IterableLookUp(Dataset):
    def __init__(self,
                 active_db,
                 cached_db=None,
                 model_emb=None,
                 answer_data_json=None,
                 top_k=25):
        self.answer_data_json = answer_data_json
        self.model_emb = model_emb
        self.top_k = top_k
        if cached_db is None:
            cached_db = active_db
        self.reorder_db(active_db, cached_db)
        return

    @staticmethod
    def custom_collate_fn(batch):
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
        NEW_BATCH['o'] = torch.nn.functional.one_hot(NEW_BATCH['o'], num_classes=6370).squeeze(1).to('cuda')
        NEW_BATCH['o'] = NEW_BATCH['o'].float()
        NEW_BATCH['n_answers'] = torch.nn.functional.one_hot(NEW_BATCH['n_answers'], num_classes=6370).squeeze(1).to(
            'cuda')
        NEW_BATCH['n_answers'] = NEW_BATCH['n_answers'].float()
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
            # Get YOUR QUESTIONS #
            for active_question_items in active_entry['mc_question']:
                # Get YOUR OPTIONS #
                active_option_frames = active_question_items['options']
                if self.answer_data_json is not None:
                    active_option_frames = torch.LongTensor(
                        [self.answer_data_json[f]['int_index'] for f in active_option_frames])
                # Get YOUR GT ANSWER (if any) #
                active_gt_frames = active_question_items['answer_id']
                # Get the same question in the cached dataset #
                cached_question_items = cached_db[active_question_items['question']]

                # Get the Video and the Audio Frames from it #
                cached_video_frames = torch.stack(cached_question_items['video_emb'], dim=0)
                cached_audio_frames = torch.stack(cached_question_items['audio_emb'], dim=0)

                # Get the Answers from it #
                cached_answer_frames = torch.LongTensor(cached_question_items['answer'])

                # Find top k video frames #
                top_k_video_frames = cached_video_frames

                # Find top k audio frames #
                top_k_audio_frames = cached_audio_frames

                # Find top k answers #
                top_k_answers = cached_answer_frames.squeeze(0)
                top_k_answers = top_k_answers

                # PAD
                # pad_length = 688 - top_k_video_frames.size()[0]
                # if pad_length > 0:
                #     for _ in range(pad_length):
                #         top_k_video_frames = torch.concat(
                #             [top_k_video_frames, torch.zeros(size=(1, top_k_video_frames.size(-1)), device='cpu')],
                #             dim=-2)
                #         top_k_audio_frames = torch.concat(
                #             [top_k_audio_frames, torch.zeros(size=(1, top_k_audio_frames.size(-1)), device='cpu')],
                #             dim=-2)
                #         top_k_answers = torch.concat(
                #             [top_k_answers, torch.zeros(size=(1,), device='cpu')])

                db.update({idx: {
                    'v': active_video_frames.squeeze(0),
                    'aud': active_audio_frames.squeeze(0),
                    'q': active_question_items['question'],
                    'o': active_option_frames,
                    'n_feats': top_k_video_frames,
                    'n_auds': top_k_audio_frames,
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


class VideoAudioFreqLearnMCVQA:
    """Multiple-Choice vQA Model using Video and Audio Inputs, Frequency  and Q/A as embedding pairs"""

    def __init__(self, active_ds, cache_ds=None, model_emb='all-MiniLM-L12-v2', look_for_one_hot=True):
        self.lfoh = look_for_one_hot
        self.embedding_transformer = SentenceTransformer(f'sentence-transformers/{model_emb}')
        self.model = MultiRetrievalAugmentedEmbeddingV2()
        self.model.to(device='cuda')
        self._active_ds = copy.deepcopy(active_ds)
        if cache_ds is not None:
            self._cache_ds = self.build_video_answer_db_2(cache_ds)
        else:
            self._cache_ds = self.build_video_answer_db_2(active_ds)

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
                    question_db[question['question']] = {'video_emb': [], 'audio_emb': [], 'answer': [],
                                                         'answer_int': [], 'options': []}

                if self.lfoh:
                    answer = self.answer_data_json[question['options'][question['answer_id']]]['int_index']
                else:
                    answer = question['options'][question['answer_id']]

                answer_int = int(question['answer_id'])
                video_emb = entry['frames'].cpu()
                audio_emb = entry['audio'].cpu()

                if self.lfoh:
                    options = [self.answer_data_json[f]['int_index'] for f in question['options']]
                else:
                    options = question['options']
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['audio_emb'].append(audio_emb)
                question_db[question['question']]['answer'].append(answer)
                question_db[question['question']]['answer_int'].append(answer_int)
                question_db[question['question']]['options'].append(options)

        return question_db

    def fit(self, lr=0.0005, bs=128, epochs=100, val_external_dataset=None):
        FIXED_BATCH_SIZE = 1
        self.model.train()
        dataset = IterableLookUp(active_db=self._active_ds,
                                 cached_db=self._cache_ds,
                                 model_emb=self.embedding_transformer,
                                 answer_data_json=self.answer_data_json)

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
                    o, \
                    n_feats, \
                    n_answ, \
                    n_auds, \
                    gt_answer_int = batch['v'], \
                    batch['aud'], \
                    batch['o'], \
                    batch['n_feats'], \
                    batch['n_answers'], \
                    batch['n_auds'], \
                    batch['gt_answer_int']
                _, loss, metric = self.model(v=v,
                                             aud=aud,
                                             o=o,
                                             n_ids=None,
                                             n_feats=n_feats,
                                             n_answ=n_answ,
                                             n_auds=n_auds,
                                             gt_answer_int=gt_answer_int)

                epoch_loss += loss.detach().item()
                epoch_metric += metric.detach().item()
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Step': real_index, 'Loss': epoch_loss / (step_idx + 1),
                     'Accuracy': epoch_metric / (step_idx + 1)})
                loss = loss / bs
                loss.backward()
                if ((step_idx + 1) % bs == 0) or (step_idx + 1 == len(dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

            # END OF EPOCH EVAL #
            #self.eval(val_external_dataset)

        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'video_audio_learnable_baseline')

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
        dataset = IterableLookUp(active_db=active_db,
                                 cached_db=self._cache_ds,
                                 model_emb=self.embedding_transformer,
                                 answer_data_json=self.answer_data_json)

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
                    o, \
                    n_feats, \
                    n_answ, \
                    n_auds, \
                    gt_answer_int = batch['v'], \
                    batch['aud'], \
                    batch['o'], \
                    batch['n_feats'], \
                    batch['n_answers'], \
                    batch['n_auds'], \
                    batch['gt_answer_int']
                _sim_scores, _, _ = self.model(v=v,
                                               aud=aud,
                                               o=o,
                                               n_ids=None,
                                               n_feats=n_feats,
                                               n_answ=n_answ,
                                               n_auds=n_auds,
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
                 option_frames=None,
                 n_ids=None,
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
        if option_frames is not None:
            no_frames = stored_response['answer']

        _frames = frames.unsqueeze(0).to('cuda')
        _audio_frames = audio_frames.unsqueeze(0).to('cuda')
        _db_frames = db_frames.unsqueeze(0).to('cuda')
        _audb_frames = audb_frames.unsqueeze(0).to('cuda')
        ### Fix incoming frames ###
        if not self.answer_data_json is None and option_frames is not None:
            o_ = torch.LongTensor([self.answer_data_json[f]['int_index'] for f in option_frames])
            o_ = torch.nn.functional.one_hot(o_, num_classes=6370).unsqueeze(0)
            o_ = o_.float()
            o_ = o_.to('cuda')
            ########
            _no_frames = torch.nn.functional.one_hot(torch.LongTensor([no_frames]), num_classes=6370)
            _no_frames = _no_frames.float()
            _no_frames = _no_frames.to('cuda')
            ### Fix incoming masks ###
            if n_ids is not None:
                _n_ids = None
        with torch.no_grad():
            _sim_scores, _, _ = self.model(
                v=_frames,
                aud=_audio_frames,
                o=o_,
                n_ids=None,
                n_feats=_db_frames,
                n_answ=_no_frames,
                n_auds=_audb_frames,
                gt_answer_int=None)
        gg = _sim_scores.squeeze()
        answer_id = [gg.argmax().item()]
        answer = [stored_response['answer'][gg.argmax().item()]]
        return answer_id, answer


def test_train_reordering(json_ds):
    def bvadb2(dataset):
        with open(ANSWER_DATA_PATH, 'r') as fin:
            answer_data_json = json.load(fin)
        question_db = {}
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                try:
                    question_db[question['question']]
                except KeyError:
                    question_db[question['question']] = {'video_emb': [],
                                                         'audio_emb': [],
                                                         'answer': [],
                                                         'answer_int': [],
                                                         'options': []}
                answer = answer_data_json[question['options'][question['answer_id']]]['int_index']
                answer_int = int(question['answer_id'])
                video_emb = entry['frames'].cpu()
                audio_emb = entry['audio'].cpu()
                options = [answer_data_json[f]['int_index'] for f in question['options']]
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['audio_emb'].append(audio_emb)
                question_db[question['question']]['answer'].append(answer)
                question_db[question['question']]['answer_int'].append(answer_int)
                question_db[question['question']]['options'].append(options)

        return question_db

    active = bvadb2(json_ds)
    dataset = DbAudLearnDataset(active_db=active,
                                cached_db=None,
                                model_emb=None)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=False,
                            collate_fn=dataset.custom_collate_fn)
    print('Stop Here')
    return
