import copy
import json
import pickle

import torch.nn.functional
import tqdm

from data import PerceptionDataset
from models.FreqBaseline import FreqMCVQABaseline, VideoFreqMCVQABaseline, VideoLangMCVQABaseline
from models.LearnableAudioBaseline import VideoAudioFreqLearnMCVQA
from models.LearnableBaseline import VideoLangLearnMCVQA, VideoFreqLearnMCVQA
from models.OcrAugmentedModel import OCRVideoAudioMCVQA
from utils import load_db_json, CAT, calc_top, calc_top_by_cat
import numpy as np

train_db_path = 'data/mc_question_train.json'
train_db_dict = load_db_json(train_db_path)
valid_db_path = 'data/mc_question_valid.json'
valid_db_dict = load_db_json(valid_db_path)
test_db_path = 'data/mc_question_test.json'
test_db_dict = load_db_json(test_db_path)

test_runs = 1
num_shots = [-1]  # 1, 5, -1]  # 0 shot is random
train_cfg = {'video_folder': './data/train/videos/',
             'task': 'mc_question',
             'split': 'train',
             'js_only': False,
             'use_audio': True,
             'use_ocr': True,
             }
train_mc_vqa_dataset = PerceptionDataset(train_db_dict, **train_cfg)
valid_cfg = {'video_folder': './data/valid/',
             'task': 'mc_question',
             'split': 'valid',
             'js_only': False,
             'use_audio': True,
             'use_ocr': True,
             }
val_mc_vqa_dataset = PerceptionDataset(valid_db_dict, **valid_cfg)
test_cfg = {'video_folder': './data/test/',
            'task': 'mc_question',
            'split': 'test',
            'js_only': False,
            'use_audio': True,
            'use_ocr': True,
            }

test_mc_vqa_dataset = PerceptionDataset(test_db_dict, **test_cfg)
# model = VideoAudioFreqLearnMCVQA(active_ds=val_mc_vqa_dataset,
#                                  cache_ds=train_mc_vqa_dataset.union(val_mc_vqa_dataset),
#                                  use_embedding=True,
#                                  use_aux_loss=0,
#                                  overconfidence_loss=None, model_version=1)
#
# model.load_weights('Model_Trained.pth.pth')
# model.model.eval()
train_mc_vqa_dataset.union(val_mc_vqa_dataset)
model = OCRVideoAudioMCVQA(active_ds=None,
                           cache_ds=train_mc_vqa_dataset,
                           use_embedding=False,
                           use_aux_loss=0,
                           model_version=4, top_k=25, train_skip_self=False)
model.model.eval()
answers = {}
for video_item in tqdm.tqdm(test_mc_vqa_dataset):
    # for video_item in val_mc_vqa_dataset:
    video_id = video_item['metadata']['video_id']
    video_answers = []

    for q_idx, q in enumerate(video_item['mc_question']):
        # video_model inference
        q_answer_id, q_answer = model.answer_q(
            frames=video_item['frames'],
            question=q,
            audio_frames=video_item['audio'],
            ocr_frames=video_item['ocr'],
            option_frames=q['options'],
            shots=-1,
            top_k=25,  # 0.6046@25/Pre 0.4
            sample=False,
            temp=1,
            use_gt_options=True,
            pre_softmax=True,
            post_softmax=False,
            approximate_gt_options=False, use_embedding=False)
        answer_dict = {
            'id': q['id'],
            'answer_id': q_answer_id[0],
            'answer': q['options'][q_answer_id[0]]
        }
        video_answers.append(answer_dict)

    answers[video_id] = video_answers

with open('test_results_static_retrieve_ocr.json', 'w') as my_file:
    json.dump(answers, my_file)
