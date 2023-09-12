import json
import tqdm

from data import PerceptionDataset
from models.SoftRetrievalEmb import SR_MCVQA_EMB
from utils import load_db_json, CLIP_Q_DATA_PATH, CLIP_A_DATA_PATH

train_db_path = 'data/mc_question_train.json'
train_db_dict = load_db_json(train_db_path)
test_db_path = 'data/mc_question_test.json'
test_db_dict = load_db_json(test_db_path)

test_runs = 1
num_shots = [-1]  # 1, 5, -1]  # 0 shot is random
train_cfg = {'video_folder': './data/train/videos/',
             'qo_folders': [[CLIP_Q_DATA_PATH, CLIP_A_DATA_PATH]],
             'task': 'mc_question',
             'split': 'train',
             'js_only': False,
             'use_audio': True,
             'use_ocr': True,
             'use_qo': True,
             }
train_mc_vqa_dataset = PerceptionDataset(train_db_dict, **train_cfg)
test_cfg = {'video_folder': './data/test/',
            'qo_folders': [[CLIP_Q_DATA_PATH, CLIP_A_DATA_PATH]],
            'task': 'mc_question',
            'split': 'test',
            'js_only': False,
            'use_audio': True,
            'use_ocr': True,
            'use_qo': True,
            }

test_mc_vqa_dataset = PerceptionDataset(test_db_dict, **test_cfg)
model = SR_MCVQA_EMB(active_ds=None,
                     cache_ds=train_mc_vqa_dataset,
                     look_for_one_hot=False,
                     use_embedding=False,
                     use_aux_loss=0,
                     model_version=5,
                     top_k=11,
                     train_skip_self=True, common_dropout=0)
model.load_weights(path='SR_MCVQA_Model5.pth', part=1)
model.load_weights(path='SR_MCVQA_EMB_Model25.pth', part=2)
model.model.eval()

answers = {}
for video_item in tqdm.tqdm(test_mc_vqa_dataset):
    # for video_item in val_mc_vqa_dataset:
    video_id = video_item['metadata']['video_id']
    video_answers = []

    for q_idx, q in enumerate(video_item['mc_question']):
        # video_model inference
        q_answer_id, _ = model.answer_q(
            frames=video_item['frames'],  # Video Instance
            question=q,  # Question Instance
            audio_frames=video_item['audio'],  # Audio Instance
            ocr_frames=video_item['ocr'],  # OCR Instance
            option_frames=video_item['clip_o_frames'],  # Options Instance
        )
        answer_dict = {
            'id': q['id'],
            'answer_id': q_answer_id[0],
            'answer': q['options'][q_answer_id[0]]
        }
        video_answers.append(answer_dict)

    answers[video_id] = video_answers

with open('test_results_test_v_train_clip.json', 'w') as my_file:
    json.dump(answers, my_file)
