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


class LateEmbFusion2Decision(nn.Module):
    def __init__(self, v_dim=762, l_dim=762, qa_paired=False):
        super().__init__()
        self.v_dim = v_dim
        self.q_dim = l_dim
        self.emb_dim = min(self.v_dim, self.q_dim)
        self.qa_paired = qa_paired
        self.v_block = nn.Sequential(nn.Linear(self.v_dim, self.emb_dim), nn.Dropout(p=0.1))
        self.q_block = nn.Sequential(nn.Linear(self.q_dim, self.emb_dim), nn.Dropout(p=0.1))
        self.a_block = nn.Sequential(nn.Linear(self.q_dim, self.emb_dim))

    def forward(self, v, q, a=None, gt=None):
        v = self.v_block(v)  # BS, E_DIM
        if self.qa_paired:
            # The q contains both the question and the answer #
            assert a is None
            q = self.q_block(q)  # BS, 3, E_DIM



        else:
            # The q contains only the question #
            a = self.a_block(a)  # BS, 3, E_DIM
            q = self.q_block(q)  # BS, E_DIM

        return


class VideoLangLearnMCVQA:
    """Multiple-Choice vQA Model using Video Inputs and Q/A as embedding pairs"""

    def __init__(self, train_dataset, model_emb='all-MiniLM-L12-v2'):
        self.video_emb_answer_db = self.build_video_answer_db(train_dataset)
        self.embedding_transformer = SentenceTransformer(f'sentence-transformers/{model_emb}')
        self.model = None

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
