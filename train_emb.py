from data import PerceptionDataset
from models.SoftRetrievalEmb import SR_MCVQA_EMB
from utils import load_db_json, CLIP_Q_DATA_PATH, CLIP_A_DATA_PATH

train_db_path = 'data/mc_question_train.json'
train_db_dict = load_db_json(train_db_path)

valid_db_path = 'data/mc_question_valid.json'
valid_db_dict = load_db_json(valid_db_path)

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
valid_cfg = {'video_folder': './data/valid/',
             'qo_folders': [[CLIP_Q_DATA_PATH, CLIP_A_DATA_PATH]],
             'task': 'mc_question',
             'split': 'valid',
             'js_only': False,
             'use_audio': True,
             'use_ocr': True,
             'use_qo': True,
             }
val_mc_vqa_dataset = PerceptionDataset(valid_db_dict, **valid_cfg)
model = SR_MCVQA_EMB(active_ds=train_mc_vqa_dataset,
                     cache_ds=train_mc_vqa_dataset,
                     look_for_one_hot=False,
                     use_embedding=False,
                     use_aux_loss=0,
                     model_version=5,
                     top_k=10,
                     train_skip_self=True)

for i in range(100):
    model.fit(lr=0.001, bs=32, epochs=10)
    if i % 2 == 1:
        model.eval(val_dataset=val_mc_vqa_dataset)
