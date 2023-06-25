import copy
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
import re
from torch.utils.data import Dataset, DataLoader


class LateEmbFusion2Decision(nn.Module):
    def __init__(self, v_dim=762, l_dim=762, qa_paired=False, smooth_labels=0.0):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = l_dim
        self.emb_dim = min(self.v_dim, self.q_dim)
        self.qa_paired = qa_paired
        self.v_block = nn.Sequential(nn.Linear(self.v_dim, self.emb_dim), nn.Dropout(p=0.1))
        self.q_block = nn.Sequential(nn.Linear(self.q_dim, self.emb_dim), nn.Dropout(p=0.1))
        self.a_block = nn.Sequential(nn.Linear(self.q_dim, self.emb_dim))
        coef = 3 if qa_paired else 2
        self.d_block = nn.Sequential(nn.Linear(self.emb_dim * coef, self.emb_dim), nn.ReLU(),
                                     nn.Linear(self.emb_dim, 1))
        self.smooth_labels = smooth_labels
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.smooth_labels)

    def calc_loss(self, scores, gt):
        """
        Calculate Cross-Entropy Loss
        :param scores: The unnormalised logits
        :param gt: A list of the true labels
        :return:
        """
        if not isinstance(gt, torch.Tensor):
            gt = torch.LongTensor(gt).to('cuda')
        return self.loss_fn(scores, gt)

    def forward(self, v, q, a=None, gt=None):
        v = self.v_block(v)  # BS, E_DIM
        if self.qa_paired:
            # The q contains both the question and the answer #
            assert a is None
            q = self.q_block(q)  # BS, 3, E_DIM

            v = v.unsqueeze(1).repeat(1, 3, 1)  # BS, 3, E_DIM
            vq = torch.cat([v, q], dim=2)  # BS, 3, 2 * E_DIM
            scores = self.d_block(vq)  # BS, 3, 1
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
            return scores, loss
        return scores, None


class DbDataset(Dataset):
    def __init__(self, video_emb_answer_db, model_emb):
        self.video_emb_answer_db = video_emb_answer_db
        self.video_emb_answer_db = self.reorder_db(qa_paired=False)
        self.model_emb = model_emb
        return

    def reorder_db(self, qa_paired=False):
        indx = 0
        db = {}
        if not qa_paired:
            corpus_question = []
            corpus_answer = []
            corpus_options = []
            for k, v in self.video_emb_answer_db:
                corpus_question.append(k)
                corpus_options.append(v['options'])
                corpus_answer.append(v['answer'])

            emb_question = self.model_emb.encode(corpus_question, batch_size=32,
                                                 convert_to_tensor=True,
                                                 device='cuda')
            emb_options = self.model_emb.encode(corpus_options, batch_size=32,
                                                convert_to_tensor=True,
                                                device='cuda')
            emb_answers = self.model_emb.encode(corpus_answer, batch_size=32,
                                                convert_to_tensor=True,
                                                device='cuda')
            for k, v in self.video_emb_answer_db:
                video = v['video_emb']
                db.update({indx: {'v': video, 'q': emb_question[indx],
                                  'a': [emb_options[indx], emb_options[indx + 1], emb_options[indx + 2]],
                                  'gt': emb_answers[indx]}})
                indx += 1
        else:
            corpus_question = []
            corpus_answer = []
            for k, v in self.video_emb_answer_db:
                corpus_question.append(k + v['options'][0])
                corpus_question.append(k + v['options'][1])
                corpus_question.append(k + v['options'][2])
                corpus_answer.append(v['answer'])

            emb_question = self.model_emb.encode(corpus_question, batch_size=32,
                                                 convert_to_tensor=True,
                                                 device='cuda')
            emb_answers = self.model_emb.encode(corpus_answer, batch_size=32,
                                                convert_to_tensor=True,
                                                device='cuda')
            for k, v in self.video_emb_answer_db:
                video = v['video_emb']
                db.update({indx: {'v': video, 'q': [emb_question[indx], emb_question[indx + 1], emb_question[indx + 2]],
                                  'gt': emb_answers[indx]}})
                indx += 1

        return db

    def __len__(self):
        return len(self.video_emb_answer_db)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.video_emb_answer_db[idx]


class VideoLangLearnMCVQA:
    """Multiple-Choice vQA Model using Video Inputs and Q/A as embedding pairs"""

    def __init__(self, train_dataset, model_emb='all-MiniLM-L12-v2'):
        self.video_emb_answer_db = self.build_video_answer_db(train_dataset)
        self.embedding_transformer = SentenceTransformer(f'sentence-transformers/{model_emb}')
        self.model = LateEmbFusion2Decision(v_dim=762, l_dim=762, qa_paired=True, smooth_labels=0.0)

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
                    question_db[question['question']] = {'video_emb': [], 'answer': []}

                answer = question['options'][question['answer_id']].lower()
                video_emb = entry['frames'].cpu()
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['answer'].append(answer)

        return question_db

    def fit(self, lr, bs, epochs):

        pass

    def answer_q(self, frames, question, top_k=None, **args):
        """
            Answer a multiple choice question.
            :param question: The question to answer
            :param frames: The incoming Video Frames
            :param top_k: Filter the top-k similar results
        """

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
            if i not in top_neighbours:
                v = 0
            candidate_answer = stored_response['answer'][i].lower()
            if candidate_answer not in sum_scores:
                sum_scores.update({candidate_answer: v})
            else:
                sum_scores[candidate_answer] += v

        candidate_options = [f.lower() for f in question['options']]
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

        answer = []
        answer_id = []
        try:
            answer.append(new_answer_list[max_pos])
            answer_id.append(candidate_options.index(new_answer_list[max_pos]))
        except:
            ra = 0
            answer.append(candidate_options[ra])
            answer_id.append(ra)
        return answer_id, answer
