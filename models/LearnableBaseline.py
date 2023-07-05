import copy
import json
import math
import os.path
import pickle
import random
from typing import Dict, Any, Tuple
import numpy as np
import sentence_transformers.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy
from sympy import divisors
from utils import ANSWER_DATA_PATH


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth')


class FakeBlock(nn.Module):
    def __init__(self, in_dim=384, out_dim=None):
        super().__init__()
        self.in_dim = in_dim
        if out_dim is None:
            out_dim = self.in_dim
        self.out_dim = out_dim

    def forward(self, input, *args):
        return torch.randn(size=(input.size(0), self.out_dim), device='cuda')


class EncoderBlock(nn.Module):
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


class StrippedEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(data=torch.ones(1, ), requires_grad=True)

    def forward(self, q, k, v):
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / (self.temp * math.sqrt(dk))
        attention = F.softmax(scores, dim=-1)
        y = attention.matmul(v)
        return y


class LateEmbFusion2Decision(nn.Module):
    def __init__(self, v_dim=768, l_dim=384, qa_paired=False, smooth_labels=0):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = l_dim
        self.emb_dim = min(self.v_dim, self.q_dim)
        self.qa_paired = qa_paired
        self.v_block = nn.Sequential(FakeBlock(self.v_dim))
        self.q_block = nn.Sequential(nn.Identity())
        self.q_block_scale = nn.Sequential(nn.Linear(self.q_dim, self.q_dim))
        self.q_block_bias = nn.Sequential(nn.Linear(self.q_dim, self.q_dim))
        self.a_block = nn.Sequential(nn.Identity())
        coef = 1 if qa_paired else 2
        heads = 3 if qa_paired else 1
        self.d_block = nn.Sequential(nn.Linear(2 * self.q_dim, self.emb_dim), nn.ReLU(),
                                     nn.Linear(self.emb_dim, 1))
        self.smooth_labels = smooth_labels
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.smooth_labels)
        self.metric_fn = MulticlassAccuracy(num_classes=3)

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
        return self.metric_fn(scores, gt)

    def forward(self, v, q=None, a=None, gt=None):
        v = self.v_block(v)  # BS, E_DIM
        if self.qa_paired:
            # The a contains both the question and the answers #
            q = self.q_block(q)
            q_emb_s = self.q_block_scale(q)  # BS, E_DIM
            q_emb_b = self.q_block_bias(q)  # BS, E_DIM
            # BS, E_DIM
            q = q.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            q_emb_s = q_emb_s.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            q_emb_b = q_emb_b.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            a = a * q_emb_s  # BS, 3, E_DIM
            a = a + q_emb_b  # BS, 3, E_DIM
            va = torch.cat([q, a], dim=2)
            scores = self.d_block(va)  # BS, 3, 1
            scores = scores.squeeze(2)  # BS, 3
        else:
            # The q contains only the question #
            a = self.a_block(a)  # BS, 3, E_DIM
            q = self.q_block(q)  # BS, E_DIM

            v = v.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            q = q.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            vqa = torch.cat([v, q, a], dim=2)  # BS, 3, 3 * E_DIM
            scores = self.d_block(vqa)  # BS, 3, 1
            scores = scores.squeeze(2)  # BS, 3

        if gt is not None:
            loss = self.calc_loss(scores, gt)
            metric = self.calc_acc(scores, gt)
            return scores, loss, metric
        return scores, None, None


class RetrievalAugmentedEmbedding(nn.Module):
    def __init__(self, v_dim=768, l_dim=384, o_dim=None):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = l_dim
        if o_dim is None:
            self.o_dim = self.q_dim
        else:
            self.o_dim = o_dim
        self.emb_dim = self.v_dim
        # Trainable Query #
        #self.tq = torch.nn.Parameter(torch.randn(size=(1, 1, self.emb_dim)), requires_grad=True)
        # Common Dropout #
        self.common_dropout = nn.Dropout1d(p=0.0)
        # Common VISION BLOCKS #
        self.common_vision_backbone = nn.Identity()
        self.v_block = nn.Sequential(self.common_vision_backbone)
        self.v2_block = nn.Sequential(self.common_vision_backbone, self.common_dropout)

        self.q_block = nn.Identity()

        # Common OPTION BLOCKS #
        # self.common_option_backbone = nn.Sequential(nn.Linear(self.o_dim, self.emb_dim))
        self.common_option_backbone = nn.Identity()
        # self.common_option_backbone = nn.Identity()
        self.o_block = nn.Sequential(self.common_option_backbone)
        self.o2_block = nn.Sequential(self.common_option_backbone, self.common_dropout)

        self.calc_block = EncoderBlock(embed_dim=self.emb_dim, num_heads=16, dropout=0.1, add_skip=False)
        self.score_block = EncoderBlock(embed_dim=self.emb_dim, num_heads=16, dropout=0.1, add_skip=False)
        self.transform_block = nn.Sequential(nn.Linear(self.v_dim + self.q_dim + self.o_dim, 128), nn.ReLU(),
                                             nn.Dropout(p=0.5), nn.Linear(128, 1))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric_fn = MulticlassAccuracy(num_classes=3)

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
        return self.metric_fn(scores, gt)

    def forward(self, v, q, o, n_ids=None, n_feats=None, n_answ=None,
                gt_answer_int=None, gt_answer_emb=None):
        #################################################################################
        v = self.v_block(v)
        #nv = self.v2_block(n_feats)
        q = self.q_block(q)
        o = self.o_block(o)
        #n_answ = self.o2_block(n_answ)
        combination1 = torch.cat([v , q , o[:, 0, :]], dim=1)
        combination2 = torch.cat([v , q , o[:, 1, :]], dim=1)
        combination3 = torch.cat([v , q , o[:, 2, :]], dim=1)
        score_0 = self.transform_block(combination1)
        score_1 = self.transform_block(combination2)
        score_2 = self.transform_block(combination3)
        ########################################
        # answer = self.calc_block(v, nv, n_answ)
        # aggr = self.score_block(answer, o, o).squeeze(1)
        #########################################
        # score_0 = aggr * o[:, 0, :]  # BS, E
        # score_0 = score_0.sum(1).unsqueeze(1)
        # score_1 = aggr * o[:, 1, :]  # BS, E
        # score_1 = score_1.sum(1).unsqueeze(1)
        # score_2 = aggr * o[:, 2, :]  # BS, E
        # score_2 = score_2.sum(1).unsqueeze(1)
        scores = torch.cat([score_0, score_1, score_2], dim=1)

        if gt_answer_int is not None:
            loss = self.calc_loss(scores, gt_answer_int)
            metric = self.calc_acc(scores, gt_answer_int)
            return scores, loss, metric
        return scores, None, None


class DbDataset(Dataset):
    def __init__(self, video_emb_answer_db, model_emb, qa_paired=False):
        self.video_emb_answer_db = video_emb_answer_db
        self.model_emb = model_emb
        self.video_emb_answer_db = self.reorder_db(qa_paired=qa_paired)
        return

    @staticmethod
    def custom_collate_fn(batch):
        KEYS = batch[0].keys()
        NEW_BATCH = {k: [] for k in KEYS}
        for entry in batch:
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    pass
                elif isinstance(v, list):
                    v = torch.stack(v, dim=0)
                elif isinstance(v, int):
                    v = torch.LongTensor([v])
                v = v.to('cuda')
                NEW_BATCH[k].append(v)

        for k in KEYS:
            NEW_BATCH[k] = torch.stack(NEW_BATCH[k], dim=0)
        return NEW_BATCH

    def reorder_db(self, qa_paired=False):
        indx = 0
        db = {}
        if not qa_paired:
            corpus_question = []
            corpus_answer = []
            corpus_options = []
            for k, v in self.video_emb_answer_db.items():
                for _ in range(len(v['answer_int'])):
                    corpus_question.append(k)
                for ii in v['options']:
                    for jj in ii:
                        corpus_options.append(jj)
                for ii in v['answer_int']:
                    corpus_answer.append(ii)
        else:
            corpus_question = []
            corpus_answer = []
            corpus_options = []
            for k, v in self.video_emb_answer_db.items():
                for _ in range(len(v['answer_int'])):
                    corpus_question.append(k)
                for ii in v['options']:
                    for jj in ii:
                        corpus_options.append(jj)
                for ii in v['answer_int']:
                    corpus_answer.append(ii)

        emb_question = self.model_emb.encode(corpus_question, batch_size=32,
                                             convert_to_tensor=True,
                                             device='cuda')
        emb_options = self.model_emb.encode(corpus_options, batch_size=32,
                                            convert_to_tensor=True,
                                            device='cuda')
        for k, v in self.video_emb_answer_db.items():
            for ii in range(len(v['answer_int'])):
                video = v['video_emb'][ii].to(device='cuda')
                db.update({indx: {'v': video, 'q': emb_question[indx],
                                  'a': [emb_options[indx], emb_options[indx + 1], emb_options[indx + 2]],
                                  'gt': corpus_answer[indx]}})
                indx += 1

        return db

    def __len__(self):
        return len(self.video_emb_answer_db)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.video_emb_answer_db[idx]


class DbLearnDataset(Dataset):
    def __init__(self, video_emb_answer_db, model_emb, video_emb_val_db=None, mode='train'):
        self.mode = mode
        self.model_emb = model_emb
        self.video_emb_answer_db = video_emb_answer_db
        if video_emb_val_db is None:
            self.video_emb_val_db = None
        else:
            print("Reordering Test DB...\n")
            self.video_emb_val_db = video_emb_val_db
            self.video_emb_val_db = self.freq_reorder_test()
        print("Reordering Train DB...\n")
        self.video_emb_answer_db = self.freq_reorder_train()
        return

    def set_train_mode(self):
        self.mode = 'train'

    def set_eval_mode(self):
        self.mode = 'eval'

    @staticmethod
    def custom_collate_fn(batch):
        KEYS = batch[0].keys()
        NEW_BATCH = {k: [] for k in KEYS}
        for entry in batch:
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    pass
                elif isinstance(v, list):
                    if isinstance(v[0], int):
                        v = torch.LongTensor([v])
                    else:
                        v = torch.stack(v, dim=0)
                elif isinstance(v, int):
                    v = torch.LongTensor([v])
                v = v.to('cuda')
                NEW_BATCH[k].append(v)

        for k in KEYS:
            NEW_BATCH[k] = torch.stack(NEW_BATCH[k], dim=0)
        # Expand Options #
        NEW_BATCH['o'] = torch.nn.functional.one_hot(NEW_BATCH['o'], num_classes=6370).squeeze(1).to('cuda')
        NEW_BATCH['o'] = NEW_BATCH['o'].float()
        NEW_BATCH['n_answers'] = torch.nn.functional.one_hot(NEW_BATCH['n_answers'], num_classes=6370).squeeze(1).to(
            'cuda')
        NEW_BATCH['n_answers'] = NEW_BATCH['n_answers'].float()
        return NEW_BATCH

    def freq_reorder_train(self, top_k=25):
        idx = 0
        db = {}
        # Pre-calc Questions #
        question_embs = self.model_emb.encode(list(self.video_emb_answer_db.keys()),
                                              batch_size=32,
                                              convert_to_tensor=True,
                                              device='cuda')

        # For each question in the DB #
        for ij, (question, stored_response) in enumerate(tqdm.tqdm(self.video_emb_answer_db.items())):
            question_emb = question_embs[ij]

            db_frames = torch.stack(stored_response['video_emb'], dim=0)
            if len(stored_response['video_emb'][0].size()) == 1:
                db_frames = torch.stack(stored_response['video_emb'], dim=0)
            elif len(stored_response['video_emb'][0].size()) == 2:
                db_frames = torch.cat(stored_response['video_emb'], dim=0)

            # Pre-calc Similarities #
            db_frames = db_frames.to('cuda')
            sim_scoress = F.cosine_similarity(db_frames[None, :], db_frames[:, None], dim=-1)
            sim_scoress = torch.clip(sim_scoress, 0, 1).cpu()

            # For each Video / Answer pair #

            # Pre-cal Answers #
            # answer_embs = self.model_emb.encode(stored_response['answer'],
            #                                     batch_size=32,
            #                                     convert_to_tensor=True,
            #                                     device='cuda')
            answer_embs = stored_response['answer']
            # options_emb = self.model_emb.encode([k for f in stored_response['options'] for k in f],
            #                                     batch_size=32,
            #                                     convert_to_tensor=True,
            #                                     device='cuda')
            options_emb = [k for f in stored_response['options'] for k in f]

            for jk, (frame, answer_int) in enumerate(
                    zip(stored_response['video_emb'], stored_response['answer_int'])):
                sim_scores = sim_scoress[jk]
                if top_k is None:
                    top_k = sim_scores.size()[-1]
                max_item_pos = sim_scores.argmax().item()
                top_neighbours = [f.item() for f in sim_scores.argsort()[-(top_k + 1):] if not f == max_item_pos]

                ### Gather Neighbour similarity Tensor ###
                current_f = frame
                current_q = question_emb
                current_o = [options_emb[jk * 3], options_emb[jk * 3 + 1], options_emb[jk * 3 + 2]]

                neighbour_ids = top_neighbours
                neighbour_feats = db_frames[top_neighbours]
                neighbour_answers = [answer_embs[f] for f in top_neighbours]
                gt_answer_int = answer_int
                gt_answer_emb = current_o[answer_int]

                # PAD
                pad_length = top_k - len(neighbour_ids)
                if pad_length > 0:
                    for _ in range(pad_length):
                        neighbour_ids.append(-1)
                        neighbour_feats = torch.concat(
                            [neighbour_feats, torch.zeros(size=(1, neighbour_feats.size(-1)), device='cuda')], dim=-2)
                        # neighbour_answers.append(torch.zeros(size=(answer_embs[0].size(-1),), device='cuda'))
                        neighbour_answers.append(0)

                db.update({idx: {
                    'v': current_f,
                    'q': current_q,
                    'o': current_o,
                    'n_ids': neighbour_ids,
                    'n_feats': neighbour_feats,
                    'n_answers': neighbour_answers,
                    'gt_answer_int': gt_answer_int,
                    'gt_answer_emb': gt_answer_emb
                }})
                idx += 1
        return db

    def freq_reorder_test(self, top_k=25):
        idx = 0
        db = {}
        # Pre-calc Questions #
        question_embs = self.model_emb.encode(list(self.video_emb_val_db.keys()),
                                              batch_size=32,
                                              convert_to_tensor=True,
                                              device='cuda').to('cpu')

        # For each question in the eval DB, find the same in the train DB #
        for ij_eval, (question_eval, stored_response_eval) in enumerate(tqdm.tqdm(self.video_emb_val_db.items())):
            # Look for the same in the Train DB #
            for ij_train, (question_train, stored_response_train) in enumerate(
                    self.video_emb_answer_db.items()):
                if question_eval == question_train:
                    db_frames_eval = torch.stack(stored_response_eval['video_emb'], dim=0).to('cpu')
                    db_frames_train = torch.stack(stored_response_train['video_emb'], dim=0).to('cpu')
                    break

            question_emb = question_embs[ij_eval]
            # Pre-calc Similarities #
            x = db_frames_eval[:, None]
            y = db_frames_train[None, :]
            sim_scoress = F.cosine_similarity(x, y, dim=-1)
            sim_scoress = torch.clip(sim_scoress, 0, 1).cpu()

            # For each Video / Answer pair #

            # Pre-cal Answers #
            # answer_embs = self.model_emb.encode(stored_response_train['answer'],
            #                                     batch_size=32,
            #                                     convert_to_tensor=True,
            #                                     device='cuda').to('cpu')
            answer_embs = stored_response_train['answer']
            # options_emb = self.model_emb.encode([k for f in stored_response_eval['options'] for k in f],
            #                                     batch_size=32,
            #                                     convert_to_tensor=True,
            #                                     device='cuda').to('cpu')
            options_emb = [k for f in stored_response_eval['options'] for k in f]

            for jk, (frame, answer_int) in enumerate(
                    zip(stored_response_eval['video_emb'], stored_response_eval['answer_int'])):
                sim_scores = sim_scoress[jk]
                if top_k is None:
                    top_k = sim_scores.size()[-1]
                max_item_pos = sim_scores.argmax().item()
                top_neighbours = [f.item() for f in sim_scores.argsort()[-(top_k + 1):] if not f == max_item_pos]

                ### Gather Neighbour similarity Tensor ###
                current_f = frame
                current_q = question_emb
                current_o = [options_emb[jk * 3], options_emb[jk * 3 + 1], options_emb[jk * 3 + 2]]

                neighbour_ids = top_neighbours
                neighbour_feats = db_frames_train[top_neighbours]
                neighbour_answers = [answer_embs[f] for f in top_neighbours]
                gt_answer_int = answer_int
                gt_answer_emb = current_o[answer_int]

                # PAD
                pad_length = top_k - len(neighbour_ids)
                if pad_length > 0:
                    for _ in range(pad_length):
                        neighbour_ids.append(-1)
                        neighbour_feats = torch.concat(
                            [neighbour_feats, torch.zeros(size=(1, neighbour_feats.size(-1)))], dim=-2)
                        # neighbour_answers.append(torch.zeros(size=(answer_embs[0].size(-1),)))
                        neighbour_answers.append(0)

                db.update({idx: {
                    'v': current_f,
                    'q': current_q,
                    'o': current_o,
                    'n_ids': neighbour_ids,
                    'n_feats': neighbour_feats,
                    'n_answers': neighbour_answers,
                    'gt_answer_int': gt_answer_int,
                    'gt_answer_emb': gt_answer_emb
                }})
                idx += 1
        return db

    def __len__(self):
        if self.mode == 'eval':
            return len(self.video_emb_val_db)
        elif self.mode == 'train':
            return len(self.video_emb_answer_db)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == 'eval':
            return self.video_emb_val_db[idx]
        elif self.mode == 'train':
            return self.video_emb_answer_db[idx]


class VideoLangLearnMCVQA:
    """Multiple-Choice vQA Model using Video Inputs and Q/A as embedding pairs"""

    def __init__(self, train_dataset, model_emb='all-MiniLM-L12-v2', qa_paired=False):
        self.qa_paired = qa_paired
        self.video_emb_answer_db = self.build_video_answer_db(train_dataset)
        self.embedding_transformer = SentenceTransformer(f'sentence-transformers/{model_emb}')
        self.model = LateEmbFusion2Decision(v_dim=768, l_dim=384, qa_paired=qa_paired, smooth_labels=0.0)
        self.model.to(device='cuda')

    def build_video_answer_db(self, dataset) -> Dict[str, Any]:
        """Builds a Video Embedding - Answer database.

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
        question_db = {}
        print("Building Dataset...\n")
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                try:
                    question_db[question['question']]
                except KeyError:
                    question_db[question['question']] = {'video_emb': [], 'answer': [], 'answer_int': [], 'options': []}

                answer = question['options'][question['answer_id']]
                answer_int = int(question['answer_id'])
                video_emb = entry['frames'].cpu()
                options = question['options']
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['answer'].append(answer)
                question_db[question['question']]['answer_int'].append(answer_int)
                question_db[question['question']]['options'].append(options)

        return question_db

    def fit(self, lr=0.001, bs=1024, epochs=100, val_external_dataset=None):
        dataset = DbDataset(video_emb_answer_db=self.video_emb_answer_db, model_emb=self.embedding_transformer,
                            qa_paired=self.qa_paired)
        dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)
        if val_external_dataset is not None:
            external_video_emb_answer_db = self.build_video_answer_db(val_external_dataset)
            val_dataset = DbDataset(video_emb_answer_db=external_video_emb_answer_db,
                                    model_emb=self.embedding_transformer)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=bs, shuffle=False, pin_memory=False,
                                        collate_fn=val_dataset.custom_collate_fn)

        optimizer = torch.optim.AdamW(
            params=[
                {'params': self.model.parameters(), 'lr': lr, 'weight_decay': 0.01}
            ],
            betas=(0.99, 0.95),
            eps=1e-8
        )
        optimizer.zero_grad()

        for epoch_idx in range(epochs):
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                real_index = (epoch_idx * len(dataloader)) + step_idx
                v, q, a, gt = batch['v'], batch['q'], batch['a'], batch['gt']
                logits, loss, metric = self.model(v=v, q=q, a=a, gt=gt)
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Step': real_index, 'Loss': loss.item(),
                     'Accuracy': metric.item()})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch_idx % 10 == 9 and val_external_dataset is not None:
                avg_loss = 0
                avg_metric = 0
                self.model.eval()
                with torch.no_grad():
                    print("[VALIDATION] ---------------")
                    for step_idx, batch in enumerate(pbar := tqdm.tqdm(val_dataloader)):
                        v, q, a, gt = batch['v'], batch['q'], batch['a'], batch['gt']
                        logits, loss, metric = self.model(v=v, q=q, a=a, gt=gt)
                        avg_loss += loss.item()
                        avg_metric += metric.item()
                        pbar.set_postfix(
                            {'Epoch': epoch_idx, 'Step': step_idx, 'Loss': avg_loss / (step_idx + 1),
                             'Accuracy': avg_metric / (step_idx + 1)})
                self.model.train()

        # Save At End of Final Epoch #
        # save_checkpoint({
        #     'state_dict': self.video_model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }, f'learnable_baseline')

        # Freeze the video_model #
        self.model.eval()
        return

    def eval(self, bs=256, external_dataset=None):
        avg_loss = 0
        avg_metric = 0
        if external_dataset is None:
            external_video_emb_answer_db = self.video_emb_answer_db
        else:
            external_video_emb_answer_db = self.build_video_answer_db(external_dataset)
        dataset = DbDataset(video_emb_answer_db=external_video_emb_answer_db, model_emb=self.embedding_transformer)
        dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False, pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)
        with torch.no_grad():
            print("[VALIDATION] ---------------")
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                v, q, a, gt = batch['v'], batch['q'], batch['a'], batch['gt']
                logits, loss, metric = self.model(v=v, q=q, a=a, gt=gt)
                avg_loss += loss.item()
                avg_metric += metric.item()
                pbar.set_postfix(
                    {'Epoch': 0, 'Step': step_idx, 'Loss': avg_loss / (step_idx + 1),
                     'Accuracy': avg_metric / (step_idx + 1)})
        return

    def answer_learnable_q(self, frames, question, **args):
        """
            Answer a multiple choice question using the learnable module
            :param question: The question to answer
            :param frames: The incoming Video Frames
        """
        ### Fix incoming frames ###
        frames = frames.to('cuda')
        if len(frames.size()) == 1:
            frames = frames.unsqueeze(0)

        ### Fix incoming question and answers###
        if self.model.qa_paired is False:
            embdq = self.embedding_transformer.encode(
                [question['question'], question['options'][0], question['options'][1], question['options'][2]],
                batch_size=4, convert_to_tensor=True, device='cuda')
            q = embdq[0].unsqueeze(0)
            a = embdq[1:, :].unsqueeze(0)
        else:
            pass

        ### Answer with the video_model ###
        with torch.no_grad():
            answer_logits, _, _ = self.model(v=frames, q=q, a=a, gt=None)

        candidate_options = [f.lower() for f in question['options']]
        ra = answer_logits.argmax().cpu().item()
        answer = candidate_options[ra]
        answer_id = ra
        return answer_id, answer


class VideoFreqLearnMCVQA:
    """Multiple-Choice vQA Model using Video Inputs, Frequency  and Q/A as embedding pairs"""

    def __init__(self, train_dataset, model_emb='all-MiniLM-L12-v2', look_for_one_hot=True):
        self.lfoh = look_for_one_hot
        self.video_emb_answer_db = self.build_video_answer_db(train_dataset)
        self.embedding_transformer = SentenceTransformer(f'sentence-transformers/{model_emb}')
        self.model = RetrievalAugmentedEmbedding(v_dim=768, l_dim=384, o_dim=6370)
        self.model.to(device='cuda')
        self.model.train()

    def build_video_answer_db(self, dataset) -> Dict[str, Any]:
        """Builds a Video Embedding - Answer database.

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
                answer_data_json = json.load(fin)
        else:
            answer_data_json = None

        question_db = {}
        print("Building Dataset...\n")
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                try:
                    question_db[question['question']]
                except KeyError:
                    question_db[question['question']] = {'video_emb': [], 'answer': [], 'answer_int': [], 'options': []}

                if self.lfoh:
                    answer = answer_data_json[question['options'][question['answer_id']]]['int_index']
                else:
                    answer = question['options'][question['answer_id']]

                answer_int = int(question['answer_id'])
                video_emb = entry['frames'].cpu()

                if self.lfoh:
                    options = [answer_data_json[f]['int_index'] for f in question['options']]
                else:
                    options = question['options']
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['answer'].append(answer)
                question_db[question['question']]['answer_int'].append(answer_int)
                question_db[question['question']]['options'].append(options)

        return question_db

    def fit(self, lr=0.0001, bs=512, epochs=100, val_external_dataset=None):
        if val_external_dataset is not None:
            if os.path.exists('ved.pkl.ds'):
                with open('ved.pkl.ds', 'rb') as fin:
                    dataset = pickle.load(fin)
            else:
                external_video_emb_answer_db = self.build_video_answer_db(val_external_dataset)
                dataset = DbLearnDataset(video_emb_answer_db=self.video_emb_answer_db,
                                         video_emb_val_db=external_video_emb_answer_db,
                                         model_emb=self.embedding_transformer)
                with open('ved.pkl.ds', 'wb') as fout:
                    pickle.dump(dataset, fout)
        else:
            if os.path.exists('ned.pkl.ds'):
                with open('ned.pkl.ds', 'rb') as fin:
                    dataset = pickle.load(fin)
            else:
                dataset = DbLearnDataset(video_emb_answer_db=self.video_emb_answer_db,
                                         video_emb_val_db=None,
                                         model_emb=self.embedding_transformer)
                with open('ned.pkl.ds', 'wb') as fout:
                    pickle.dump(dataset, fout)
        dataset.set_train_mode()
        dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=True, pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)

        if val_external_dataset is not None:
            val_dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False, pin_memory=False,
                                        collate_fn=dataset.custom_collate_fn)

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        optimizer.zero_grad()

        for epoch_idx in range(epochs):
            epoch_loss = 0
            epoch_metric = 0
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                real_index = (epoch_idx * len(dataloader)) + step_idx
                v, q, o, n_ids, n_feats, n_answ, gt_answer_int, gt_answer_emb = batch['v'], batch['q'], batch['o'], \
                    batch['n_ids'], batch[
                    'n_feats'], batch['n_answers'], batch['gt_answer_int'], batch['gt_answer_emb']
                logits, loss, metric = self.model(v=v, q=q, o=o, n_ids=n_ids, n_feats=n_feats, n_answ=n_answ,
                                                  gt_answer_int=gt_answer_int, gt_answer_emb=gt_answer_emb)
                epoch_loss += loss.detach().item()
                epoch_metric += metric.detach().item()
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Step': real_index, 'Loss': epoch_loss / (step_idx + 1),
                     'Accuracy': epoch_metric / (step_idx + 1)})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch_idx % 2 == 1 and val_external_dataset is not None:
                avg_loss = 0
                avg_metric = 0
                self.model.eval()
                dataset.set_eval_mode()
                with torch.no_grad():

                    print("[VALIDATION] ---------------")
                    for step_idx, batch in enumerate(pbar := tqdm.tqdm(val_dataloader)):
                        v, q, o, n_ids, n_feats, n_answ, gt_answer_int, gt_answer_emb = batch['v'], batch['q'], batch[
                            'o'], \
                            batch['n_ids'], batch[
                            'n_feats'], batch['n_answers'], batch['gt_answer_int'], batch['gt_answer_emb']
                        logits, loss, metric = self.model(v=v, q=q, o=o, n_ids=n_ids, n_feats=n_feats, n_answ=n_answ,
                                                          gt_answer_int=gt_answer_int, gt_answer_emb=gt_answer_emb)
                        avg_loss += loss.detach().item()
                        avg_metric += metric.detach().item()
                        pbar.set_postfix(
                            {'Epoch': epoch_idx, 'Step': step_idx, 'Loss': avg_loss / (step_idx + 1),
                             'Accuracy': avg_metric / (step_idx + 1)})
                dataset.set_train_mode()
                self.model.train()

        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'learnable_baseline')

        # Freeze the video_model #
        self.model.eval()
        return

    def eval(self, bs=256, external_dataset=None):
        avg_loss = 0
        avg_metric = 0
        if external_dataset is None:
            external_video_emb_answer_db = self.video_emb_answer_db
        else:
            external_video_emb_answer_db = self.build_video_answer_db(external_dataset)
        dataset = DbDataset(video_emb_answer_db=external_video_emb_answer_db, model_emb=self.embedding_transformer)
        dataloader = DataLoader(dataset=dataset, batch_size=bs, shuffle=False, pin_memory=False,
                                collate_fn=dataset.custom_collate_fn)
        with torch.no_grad():
            print("[VALIDATION] ---------------")
            for step_idx, batch in enumerate(pbar := tqdm.tqdm(dataloader)):
                v, q, a, gt = batch['v'], batch['q'], batch['a'], batch['gt']
                logits, loss, metric = self.model(v=v, q=q, a=a, gt=gt)
                avg_loss += loss.item()
                avg_metric += metric.item()
                pbar.set_postfix(
                    {'Epoch': 0, 'Step': step_idx, 'Loss': avg_loss / (step_idx + 1),
                     'Accuracy': avg_metric / (step_idx + 1)})
        return

    def answer_q(self, frames, question, sample=False,
                 shots=0, top_k=None, temp=1, approximate_gt_options=False):
        """
            Answer a multiple choice question.
            :param sample: Whether to sample or use argmax
            :param top_k: Filter the top-k similar results
            :param temp: Use temp when sampling
        """
        assert shots >= -1

        ### Look in the DB for the same question ###
        stored_response = self.video_emb_answer_db[question['question']]
        ### Rank answers based on incoming Frame ###
        db_frames = torch.stack(stored_response['video_emb'], dim=0)

        if len(stored_response['video_emb'][0].size()) == 1:
            db_frames = torch.stack(stored_response['video_emb'], dim=0)
        elif len(stored_response['video_emb'][0].size()) == 2:
            db_frames = torch.cat(stored_response['video_emb'], dim=0)

        ### Fix incoming frames ###
        frames = frames.to('cpu')
        if len(frames.size()) == 1:
            frames = frames.unsqueeze(0)
        normalised_frames = torch.nn.functional.normalize(frames, p=2, dim=-1)
        normalised_db_frames = torch.nn.functional.normalize(db_frames, p=2, dim=-1)
        sim_scores = F.cosine_similarity(normalised_frames, normalised_db_frames)
        sim_scores = torch.clip(sim_scores, 0, 1)

        ### Use Top-K neighbours
        if top_k is None:
            top_k = sim_scores.size()[-1]
        top_neighbours = [f.item() for f in sim_scores.argsort()[-(top_k + 1):]]
        sum_scores = {}
        for i, v in enumerate(sim_scores):
            ### Drop Top-1 MAX ###
            ### TODO: Remove at Testing ###
            # if i == max_item_pos:
            #     v = 0
            if i not in top_neighbours:
                v = 0
            ### ---------------------- ###
            candidate_answer = stored_response['answer'][i].lower()
            if candidate_answer not in sum_scores:
                sum_scores.update({candidate_answer: v})
            else:
                sum_scores[candidate_answer] += v

        candidate_options = [f.lower() for f in question['options']]

        if approximate_gt_options:
            for k in candidate_options:
                if k not in sum_scores:
                    emb_k = self.sentence_transformer.encode(k, batch_size=1, convert_to_tensor=True,
                                                             device='cuda').cpu()
                    emb_known_responses = self.sentence_transformer.encode(list(sum_scores.keys()), batch_size=32,
                                                                           convert_to_tensor=True,
                                                                           device='cuda').cpu()
                    sim_scores_coef = F.softmax(sentence_transformers.util.cos_sim(emb_k, emb_known_responses))
                    sim_scores_score = sim_scores_coef * torch.Tensor([v for _, v in sum_scores.items()])
                    sum_scores.update({k: sim_scores_score.sum()})

        new_answer_list = []
        new_answer_logits = []
        max_pos = -1
        max_val = -1
        idx = 0

        for k, v in sum_scores.items():
            if k in candidate_options:
                new_answer_list.append(k)
                new_answer_logits.append(v)
                if v > max_val:
                    max_val = v
                    max_pos = idx
                idx += 1

        if sample:
            p = F.softmax(torch.Tensor(new_answer_logits) / temp, dim=-1)
            dist = torch.distributions.categorical.Categorical(probs=p, logits=None,
                                                               validate_args=None)
            if shots == 0:
                shots = 1  # Compatibility Fix
                answer_idx = [dist.sample(sample_shape=(shots,)).item()]
            else:
                answer_idx = dist.sample(sample_shape=(shots,))
        else:
            answer_idx = [max_pos]

        answer = []
        answer_id = []
        for a in answer_idx:
            try:
                answer.append(new_answer_list[a])
                answer_id.append(candidate_options.index(new_answer_list[a]))
            except:
                ra = 0
                answer.append(candidate_options[ra])
                answer_id.append(ra)
        return answer_id, answer
