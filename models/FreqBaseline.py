import random
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


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
                 shots: int) -> Tuple[int, str]:
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

    def __init__(self, train_dataset):
        self.video_emb_answer_db = self.build_video_answer_db(train_dataset)

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

                answer = question['options'][question['answer_id']]
                video_emb = entry['frames'].cpu()
                question_db[question['question']]['video_emb'].append(video_emb)
                question_db[question['question']]['answer'].append(answer)

        return question_db

    def visual_random_answer(self, frames: torch.Tensor, question: Dict[str, Any]) -> Tuple[int, str]:
        """Answer a mc question with a visually-informed random valid answer.

        Picks a random answer for a multiple choice question.

        Args:
          question (Dict): Question.

        Returns:
          Tuple: Answer ID and answer text.
        """
        ### Look in the DB for the same question ###
        stored_response = self.video_emb_answer_db[question['question']]
        ### Rank answers based on incoming Frame ###
        db_frames = torch.cat(stored_response['video_emb'], dim=0)
        sim_scores = F.cosine_similarity(frames, db_frames)
        sum_scores = {}
        for i, v in enumerate(sim_scores):
            if stored_response['answer'][i] not in sum_scores:
                sum_scores.update({stored_response['answer'][i]: v})
            else:
                sum_scores[stored_response['answer'][i]] += v

        new_answer_list = []
        new_answer_logits = []
        for k, v in sum_scores.items():
            new_answer_list.append(k)
            new_answer_logits.append(v)

        dist = torch.distributions.categorical.Categorical(probs=None, logits=torch.Tensor(new_answer_logits),
                                                           validate_args=None)
        answer_idx = dist.sample(sample_shape=(1,)).item()
        answer = new_answer_list[answer_idx]
        answer_id = question['options'].index(answer)
        return answer_id, answer

    def answer_q(self, frames: torch.Tensor, question: Dict[str, Any],
                 shots: int) -> Tuple[int, str]:
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
        assert shots >= -1

        # sample answers to current question from stored db
        answer_sample = self.video_emb_answer_db[question['question']]

        if shots == 0:
            return self.visual_random_answer(frames, question)
        elif shots == -1 or shots > len(answer_sample):
            sample_size = len(answer_sample)
        else:
            sample_size = shots

        # # Get subsample
        # answer_subsample = random.sample(answer_sample, sample_size)
        #
        # # count frequency of valid answers
        # option_freq = []
        # for option in question['options']:
        #     option_freq.append(answer_subsample.count(option))
        #
        # # check if more than one answer has same occurence
        # max_val = max(option_freq)
        # indices = [i for i, value in enumerate(option_freq) if value == max_val]
        #
        # # if true sample from the indices
        # if len(indices) > 1:
        #     answer_id = random.sample(indices, 1)[0]
        # else:
        #     answer_id = indices[0]

        #answer = question['options'][answer_id]
        #return answer_id, answer
