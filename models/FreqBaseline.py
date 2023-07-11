import copy
import json
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

ANSWER_DATA_PATH = './data/answer_keys.json'


class NoBrainEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Parameter(data=0.4 * torch.ones(1, ), requires_grad=True)

    def forward(self, q1, k1, q2, k2, **args):
        a = self.temp
        q1 = torch.nn.functional.normalize(q1, p=2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, p=2, dim=-1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, p=2, dim=-1)
        scores1 = torch.nn.functional.cosine_similarity(q1.unsqueeze(1), k1, dim=-1)
        scores1 = torch.clip(scores1, 0, 1)
        scores2 = torch.nn.functional.cosine_similarity(q2.unsqueeze(1), k2, dim=-1)
        scores2 = torch.clip(scores2, 0, 1)
        attention1 = F.softmax(scores1, dim=-1)
        attention2 = F.softmax(scores2, dim=-1)
        attention = attention1 * a + (1 - a) * attention2
        return attention


class FreqMCVQABaseline:
    """Multiple-Choice vQA Model (Frequency Baseline)"""

    def __init__(self, db_dict: Dict[str, Any]):
        self.answer_db = self.build_answer_db(db_dict)

    def build_answer_db(self, db_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Builds an answer database.

        Returns a dictionary where the key is the question string and the value
        is a list of strings, each string a correct answer to a previous
        instance of the question.

        Args:
            db_dict (Dict): Dictionary containing the database of
                questions and corresponding answers.

        Returns:
            Dict[str, Any]: Dictionary where the key is question string and
            the value is list of full correct answers each time that question
            was asked in the db_dict.
        """
        question_db = {}
        for vid in db_dict.values():
            for question in vid['mc_question']:
                try:
                    question_db[question['question']]
                except KeyError:
                    question_db[question['question']] = []

                answer = question['options'][question['answer_id']]
                question_db[question['question']].append(answer)

        return question_db

    def random_answer(self, question: Dict[str, Any]) -> Tuple[int, str]:
        """Answer a mc question with a random valid answer.

        Picks a random answer for a multiple choice question.

        Args:
          question (Dict): Question.

        Returns:
          Tuple: Answer ID and answer text.
        """
        answer = random.sample(question['options'], 1)[0]
        answer_id = question['options'].index(answer)
        return answer_id, answer

    def answer_q(self, frames: np.array, question: Dict[str, Any],
                 shots: int, **args) -> Tuple[int, str]:
        """Answer a multiple choice question.

        Given a number of shots n, we sample n examples of correct answers, filter
        by valid answers to current question and then attempt to use the most
        common as an answer to the current question. If that is not a valid choice
        we use a random valid option as an answer." --> "Given a number of shots n,
        we sample at random n correct answers from the training set, then we count
        the number of occurrences for each of the options of the given question
        and keep the most popular answer (the answer with the highest count).
        For multiple equally popular options, we sample one at random.

        Args:
          frames: Video frames (placeholder, unused).
          question (Dict): Question.
          shots: Number of shots for multi-shot answer selection. If 0, random
          sampling is used, if -1, full population is using, else if n, uses sample
          size n of population.

        Returns:
          Tuple: Answer ID and answer text.

        """
        del frames  # unused
        assert shots >= -1

        # sample answers to current question from stored db
        answer_sample = self.answer_db[question['question']]

        if shots == 0:
            return self.random_answer(question)
        elif shots == -1 or shots > len(answer_sample):
            sample_size = len(answer_sample)
        else:
            sample_size = shots

        # get subsample
        answer_subsample = random.sample(answer_sample, sample_size)

        # count frequency of valid answers
        option_freq = []
        for option in question['options']:
            option_freq.append(answer_subsample.count(option))

        # check if more than one answer has same occurence
        max_val = max(option_freq)
        indices = [i for i, value in enumerate(option_freq) if value == max_val]

        # if true sample from the indices
        if len(indices) > 1:
            answer_id = random.sample(indices, 1)[0]
        else:
            answer_id = indices[0]

        answer = question['options'][answer_id]
        return answer_id, answer


class VideoFreqMCVQABaseline:
    """Multiple-Choice vQA Model (Frequency Baseline with Video Assistance)"""

    def __init__(self, train_dataset, model_sim='all-MiniLM-L12-v2'):
        self.lfoh = True
        self.video_emb_answer_db = self.build_video_answer_db_2(train_dataset)
        self.sentence_transformer = SentenceTransformer(f'sentence-transformers/{model_sim}')
        self.model = NoBrainEncoderBlock()
        self.model.to('cuda')


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
                    question_db[question['question']] = {'video_emb': [], 'audio_emb': [], 'answer': []}

                answer = question['options'][question['answer_id']]
                video_emb = entry['frames'].cpu()
                audio_emb = entry['audio'].cpu()
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['audio_emb'].append(audio_emb)
                question_db[question['question']]['answer'].append(answer)

        return question_db

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

    @staticmethod
    def repmore(string):
        pattern1 = r'more than (\d+)'
        pattern2 = r'(\d+) or more'

        def replace_match(match):
            number = int(match.group(1))
            return f"{number + 1} or {number + 2}"
            # return f"{number + 1}"

        replaced_string = re.sub(pattern1, replace_match, string)
        replaced_string = re.sub(pattern2, replace_match, replaced_string)

        return replaced_string

    @staticmethod
    def repless(string):
        pattern1 = r'less than (\d+)'
        pattern2 = r'(\d+) or less'

        def replace_match(match):
            number = int(match.group(1))
            return f"{number - 1} or {number - 2}"
            # return f"{number - 1}"

        replaced_string = re.sub(pattern1, replace_match, string)
        replaced_string = re.sub(pattern2, replace_match, replaced_string)

        return replaced_string

    def answer_q(self, a,
                 frames,
                 question,
                 audio_frames=None,
                 shots=0,
                 top_k=None,
                 temp=1,
                 pre_softmax=False,
                 post_softmax=False,
                 approximate_gt_options=False, **args):
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

        if len(stored_response['video_emb'][0].size()) == 1:
            db_frames = torch.stack(stored_response['video_emb'], dim=0)
        elif len(stored_response['video_emb'][0].size()) == 2:
            db_frames = torch.cat(stored_response['video_emb'], dim=0)
        if audio_frames is not None:
            if len(stored_response['audio_emb'][0].size()) == 1:
                audb_frames = torch.stack(stored_response['audio_emb'], dim=0)
            elif len(stored_response['audio_emb'][0].size()) == 2:
                audb_frames = torch.cat(stored_response['audio_emb'], dim=0)

        _frames = frames.unsqueeze(0).to('cpu')
        _audio_frames = audio_frames.unsqueeze(0).to('cpu')
        _db_frames = db_frames.unsqueeze(0).to('cpu')
        _audb_frames = audb_frames.unsqueeze(0).to('cpu')
        ### Fix incoming frames ###
        with torch.no_grad():
            _sim_scores = self.model(
                q1=_frames,
                k1=_db_frames,
                q2=_audio_frames,
                k2=_audb_frames)
        sim_scores = _sim_scores.squeeze()

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
            candidate_answer = stored_response['answer'][i]
            if candidate_answer not in sum_scores:
                sum_scores.update({candidate_answer: v})
            else:
                sum_scores[candidate_answer] += v
        if self.answer_data_json is None:
            candidate_options = [f for f in question['options']]
        else:
            candidate_options = [self.answer_data_json[f]['int_index'] for f in question['options']]


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


class VideoLangMCVQABaseline:
    """Multiple-Choice vQA Model (Frequency Baseline with Video Assistance and Language Similarity)"""

    def __init__(self, train_dataset, similarity_factor_store=0.995, similarity_factor_retrieve=0.995,
                 model_sim='all-MiniLM-L12-v2'):
        self.sfs = similarity_factor_store
        self.sfr = similarity_factor_retrieve
        self.video_lang_emb_answer_db = {}
        self.semantic_keys = None
        self.sentence_transformer = SentenceTransformer(f'sentence-transformers/{model_sim}')

        if os.path.exists('cached_pickle.pkl'):
            with open('cached_pickle.pkl', 'rb') as fin:
                data = pickle.load(fin)
                self.video_lang_emb_answer_db = data['db']
                self.semantic_keys = data['sk']
        else:
            self.build_video_lang_emb_answer_db(train_dataset)
            with open('cached_pickle.pkl', 'wb') as fout:
                data = {'db': self.video_lang_emb_answer_db, 'sk': self.semantic_keys}
                pickle.dump(data, fout)

        self.debug_counter = 0

    def _merge_shot_dicts(self, list_of_response_dicts, list_of_sim_coefs):
        signature_response = {
            'video_emb': [],
            'answer': [],
            'question_emb_idx': [],
            'question_raw': [],
            'lang_sim_coef': []
        }
        if len(list_of_response_dicts) == 0 and len(list_of_sim_coefs) == 0:
            return None
        for d, c in zip(list_of_response_dicts, list_of_sim_coefs):
            for f in d['video_emb']:
                signature_response['video_emb'].append(f)
            for f in d['answer']:
                signature_response['answer'].append(f)
            for f in d['question_emb_idx']:
                signature_response['question_emb_idx'].append(f)
            for f in d['question_raw']:
                signature_response['question_raw'].append(f)
            for _ in range(len(d['question_raw'])):
                signature_response['lang_sim_coef'].append(c)

        return signature_response

    def _db_store(self, idx, emb, entry, question):
        """
        Stores an item
        :param idx: The external entry index
        :param db: The database
        :param emb: The incoming question embedding
        :param items: The items to append to the database
        :return: Nothing
        """
        if len(self.video_lang_emb_answer_db) == 0:
            self.video_lang_emb_answer_db[0] = {'video_emb': [], 'answer': [], 'question_emb_idx': [],
                                                'question_raw': []}
            answer = question['options'][question['answer_id']]
            video_emb = entry['frames'].cpu()
            self.video_lang_emb_answer_db[0]['video_emb'].append(video_emb)
            self.video_lang_emb_answer_db[0]['answer'].append(answer)
            self.video_lang_emb_answer_db[0]['question_raw'].append(question['question'])
            self.video_lang_emb_answer_db[0]['question_emb_idx'].append(idx)
        else:
            for k, v in self.video_lang_emb_answer_db.items():
                # Avoid duplicate insertions #
                if idx not in self.video_lang_emb_answer_db[k]['question_emb_idx']:
                    sim_score = sentence_transformers.util.cos_sim(emb,
                                                                   self.semantic_keys[v['question_emb_idx'],
                                                                   :]).mean().item()
                    if sim_score >= self.sfs:
                        answer = question['options'][question['answer_id']]
                        video_emb = entry['frames'].cpu()
                        self.video_lang_emb_answer_db[k]['video_emb'].append(video_emb)
                        self.video_lang_emb_answer_db[k]['answer'].append(answer)
                        self.video_lang_emb_answer_db[k]['question_raw'].append(question['question'])
                        self.video_lang_emb_answer_db[k]['question_emb_idx'].append(idx)
                        return
                else:
                    # It already exists somewhere no point of insertion #
                    return
            # If you reached this point it means you did not insert, so we will make a new entry to the DB
            new_idx = len(self.video_lang_emb_answer_db)
            self.video_lang_emb_answer_db[new_idx] = {'video_emb': [], 'answer': [], 'question_emb_idx': [],
                                                      'question_raw': []}
            answer = question['options'][question['answer_id']]
            video_emb = entry['frames'].cpu()
            self.video_lang_emb_answer_db[new_idx]['video_emb'].append(video_emb)
            self.video_lang_emb_answer_db[new_idx]['answer'].append(answer)
            self.video_lang_emb_answer_db[new_idx]['question_raw'].append(question['question'])
            self.video_lang_emb_answer_db[new_idx]['question_emb_idx'].append(idx)
            return

    def _db_retrieve(self, question=None, emb=None, retry=False, shots=0):
        """
        Retrieves an item from the database.
        :param question: The question to be converted to an embedding.
        :param emb: The incoming question embedding.
        :param shots: The number of shots to average
        :return: The database entry
        """
        return_db_entries = []
        return_db_sim_scores = []
        if shots == 0:
            shots = 1

        best_sfr = -10
        if question is None and emb is None:
            raise ValueError("Question and Embedding cannot be both None in _db_retrieve")
        elif question is not None:
            emb = self.sentence_transformer.encode(question, convert_to_tensor=True, batch_size=1,
                                                   normalize_embeddings=False)
        else:
            assert isinstance(emb, torch.Tensor)

        for k, v in self.video_lang_emb_answer_db.items():
            sim_score = sentence_transformers.util.cos_sim(emb,
                                                           self.semantic_keys[v['question_emb_idx'], :]).mean().item()
            if sim_score >= self.sfr:
                return_db_entries.append(self.video_lang_emb_answer_db[k])
                return_db_sim_scores.append(sim_score)
                if len(return_db_entries) == shots:
                    return self._merge_shot_dicts(return_db_entries, return_db_sim_scores)
            else:
                best_sfr = max(best_sfr, sim_score)
        # If you reached this point, you did not find any match above the threshold #
        temp_sfr = best_sfr - 1e-6
        if retry:
            while temp_sfr > 0:
                for k, v in self.video_lang_emb_answer_db.items():
                    sim_score = sentence_transformers.util.cos_sim(emb,
                                                                   self.semantic_keys[v['question_emb_idx'],
                                                                   :]).mean().item()
                    if sim_score >= temp_sfr:
                        return_db_entries.append(self.video_lang_emb_answer_db[k])
                        return_db_sim_scores.append(sim_score)
                        if len(return_db_entries) == shots:
                            return self._merge_shot_dicts(return_db_entries, return_db_sim_scores)
                temp_sfr -= 0.1
        else:
            return self._merge_shot_dicts(return_db_entries, return_db_sim_scores)

    def build_video_lang_emb_answer_db(self, dataset):
        """Builds a Video Embedding - Mock Language Answer database.

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
        corpus = []

        print("Building Dataset...\n")
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                corpus.append(question['question'])

        self.semantic_keys = self.sentence_transformer.encode(corpus, convert_to_tensor=True, batch_size=32,
                                                              normalize_embeddings=False).cpu()
        total_index = 0
        for entry in tqdm.tqdm(dataset):
            for question in entry['mc_question']:
                emb = self.sentence_transformer.encode(question['question'], convert_to_tensor=True, batch_size=1,
                                                       normalize_embeddings=False).cpu()
                self._db_store(idx=total_index, emb=emb, entry=entry, question=question)
                total_index += 1
        return

    def answer_q(self, frames, question, sample=False,
                 shots=0, top_k=None, temp=1, use_gt_options=True, approximate_gt_options=False):
        """
            Answer a multiple choice question.
            :param sample: Whether to sample or use argmax
            :param top_k: Filter the top-k similar results
            :param temp: Use temp when sampling
        """
        assert shots >= -1
        ### Convert question to sentence embedding ###
        question_emb = self.sentence_transformer.encode(question['question'], convert_to_tensor=True,
                                                        normalize_embeddings=False).cpu()
        ### Look in the DB for the same semantic question ###
        stored_response = self._db_retrieve(emb=question_emb, retry=False, shots=shots)

        if len(stored_response['video_emb'][0].size()) == 1:
            db_frames = torch.stack(stored_response['video_emb'], dim=0)
        elif len(stored_response['video_emb'][0].size()) == 2:
            db_frames = torch.cat(stored_response['video_emb'], dim=0)

        ### Fix incoming frames ###
        frames = frames.to('cpu')
        if len(frames.size()) == 1:
            frames = frames.unsqueeze(0)
        sim_scores = F.cosine_similarity(frames, db_frames)
        sim_scores = torch.clip(sim_scores, 0, 1)

        ### Use Top-K neighbours
        if top_k is None:
            top_k = sim_scores.size()[-1]
        top_neighbours = [f.item() for f in sim_scores.argsort()[-(top_k + 1):]]
        max_item_pos = sim_scores.argmax().item()
        sum_scores = {}
        for i, v in enumerate(sim_scores):
            ### Drop Top-1 MAX ###
            ### TODO: Remove at Testing ###
            # if i == max_item_pos:
            #     v = 0
            if i not in top_neighbours:
                v = 0
            ### ---------------------- ###
            if stored_response['answer'][i] not in sum_scores:
                sum_scores.update({stored_response['answer'][i]: v})
            else:
                sum_scores[stored_response['answer'][i]] += v

        new_answer_list = []
        new_answer_logits = []
        max_pos = -1
        max_val = -1
        idx = 0

        for k, v in sum_scores.items():
            if use_gt_options:
                if k in question['options']:
                    new_answer_list.append(k)
                    new_answer_logits.append(v)
                    if v > max_val:
                        max_val = v
                        max_pos = idx
                    idx += 1
            else:
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
                answer_id.append(question['options'].index(new_answer_list[a]))
            except:
                if not use_gt_options:
                    ra = random.choice([0, 1, 2])
                    answer.append(question['options'][ra])
                    answer_id.append(ra)
                else:
                    # print("You chose a non-existent option for an answer!")
                    ra = random.choice([0, 1, 2])
                    answer.append(question['options'][ra])
                    answer_id.append(ra)
        return answer_id, answer
