import copy
import json

import torch.nn.functional
import tqdm

from data import PerceptionDataset
from models.FreqBaseline import FreqMCVQABaseline, VideoFreqMCVQABaseline, VideoLangMCVQABaseline
from models.LearnableBaseline import VideoLangLearnMCVQA, VideoFreqLearnMCVQA
from utils import load_db_json, CAT, calc_top, calc_top_by_cat
import numpy as np

train_db_path = 'data/mc_question_train.json'
train_db_dict = load_db_json(train_db_path)

valid_db_path = 'data/mc_question_valid.json'
valid_db_dict = load_db_json(valid_db_path)

test_runs = 1
num_shots = [-1]  # 1, 5, -1]  # 0 shot is random
train_cfg = {'video_folder': './data/train/videos/',
             'task': 'mc_question',
             'split': 'train',
             'js_only': False}
train_mc_vqa_dataset = PerceptionDataset(train_db_dict, **train_cfg)
valid_cfg = {'video_folder': './data/valid/',
             'task': 'mc_question',
             'split': 'valid',
             'js_only': False}
val_mc_vqa_dataset = PerceptionDataset(valid_db_dict, **valid_cfg)
#video_model = VideoLangMCVQABaseline(train_mc_vqa_dataset)
#video_model = VideoFreqMCVQABaseline(train_mc_vqa_dataset)
model = VideoFreqLearnMCVQA(train_mc_vqa_dataset)
model.fit(val_external_dataset=val_mc_vqa_dataset, lr=0.001)
#video_model.eval(external_dataset=val_mc_vqa_dataset)

#video_model = FreqMCVQABaseline(train_db_dict)

# results = {}
#
# for shots in num_shots:
#     test_results = []
#     test_results_dict = {k: v for k, v in zip(CAT, np.zeros((len(CAT))))}
#
#     for run in range(test_runs):
#         answers = {}
#
#         for video_item in tqdm.tqdm(val_mc_vqa_dataset):
#             # for video_item in val_mc_vqa_dataset:
#             video_id = video_item['metadata']['video_id']
#             video_answers = []
#
#             for q_idx, q in enumerate(video_item['mc_question']):
#                 # video_model inference
#                 q_answer_id, q_answer = video_model.answer_q(video_item['frames'],
#                                                        q,
#                                                        shots=shots,
#                                                        top_k=25,  # 0.6024@25
#                                                        sample=False,
#                                                        temp=1,
#                                                        use_gt_options=True, approximate_gt_options=False)
#                 answer_dict = {
#                     'id': q['id'],
#                     'answer_id': q_answer_id[0],
#                     'answer': q_answer[0]
#                 }
#                 video_answers.append(answer_dict)
#             answers[video_id] = video_answers
#
#         # calc overall top-1
#         top1 = calc_top(answers, valid_db_dict, k=1)
#         # calc top-1 by area, reasoning and tag
#         run_results_dict = calc_top_by_cat(answers, valid_db_dict)
#         for cat, cat_score in run_results_dict.items():
#             test_results_dict[cat] += cat_score
#
#         test_results.append(top1)
#     test_results_dict = {k: v / test_runs for k, v in test_results_dict.items()}
#     results[f'{shots}'] = test_results_dict
#     print(f'Results - shots: {shots} | ' +
#           f'top-1: {np.mean(test_results):.4f} | ' +
#           f'std: {np.std(test_results):.4f}')
#
#     with open('valid_results_full.json', 'w') as my_file:
#         json.dump(answers, my_file)
