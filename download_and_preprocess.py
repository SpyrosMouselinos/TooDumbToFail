import os
import pickle
import time

import torch
import tqdm
from data import PerceptionDataset
from utils import load_db_json, get_video_frames
from utils import test_download_samples
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Download annotations + train_videos
# test_download_samples()

# Constants


train_db_path = 'data/mc_question_valid.json'
pt_db_dict = load_db_json(train_db_path)

video_folder = './data/valid/'
task = 'mc_question'
split = 'valid'
N_SEGMENTS = 1
MINI_BATCH_SIZE = 16
STORE_BATCH_SIZE = 12


def load_dataset(d):
    l = []
    for _, v in d.items():
        if v['metadata']['split'] == split:
            if v[task]:
                l.append(v)
    return l


pt_db_list = load_dataset(pt_db_dict)
total_length = len(pt_db_list)
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model.classifier = nn.Identity()
model = model.to('cuda').half()


def get_video_frames_wrapper(data_items):
    loaded_videos = []
    for i in range(STORE_BATCH_SIZE):
        vid_frames = get_video_frames(data_items[i], video_folder, override_video_name=False, resize_to=224, num_samples=16, n_segments=N_SEGMENTS)
        for s in range(N_SEGMENTS):
            for f in vid_frames[s]:
                loaded_videos.append(f)

    inputs = processor(loaded_videos, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].resize(STORE_BATCH_SIZE * N_SEGMENTS, 16,
                                                           inputs['pixel_values'].size()[2],
                                                           inputs['pixel_values'].size()[3],
                                                           inputs['pixel_values'].size()[4])
    final_logits = []
    if N_SEGMENTS >= MINI_BATCH_SIZE:
        for i in range(N_SEGMENTS // MINI_BATCH_SIZE):
            semi_input = inputs['pixel_values'][i * MINI_BATCH_SIZE: (i + 1) * MINI_BATCH_SIZE].to('cuda')
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    outputs = model(pixel_values=semi_input)
                    final_logits = outputs.logits
                    final_logits = final_logits.reshape(STORE_BATCH_SIZE, N_SEGMENTS, -1)
    else:
        semi_input = inputs['pixel_values'].to('cuda')
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                outputs = model(pixel_values=semi_input)
                final_logits = outputs.logits
                final_logits = final_logits.reshape(STORE_BATCH_SIZE, N_SEGMENTS, -1)
    vid_frames = final_logits.mean(1)
    return vid_frames


def maybe_get_video_frames(data_items, n_segments=1):
    for i in range(len(data_items)):
        video_file_pickle = os.path.join(video_folder,
                                         data_items[i]['metadata']['video_id']) + f'_seg_{n_segments}.pkl'
        if os.path.exists(video_file_pickle):
            continue
        else:
            vid_frames = get_video_frames_wrapper(data_items=data_items)
            for j in range(len(data_items)):
                video_file_pickle = os.path.join(video_folder,
                                                 data_items[j]['metadata']['video_id']) + f'_seg_{n_segments}.pkl'
                with open(video_file_pickle, 'wb') as fout:
                    pickle.dump(vid_frames[j], fout)
            return
    return

s = time.time()
for batch_idx in tqdm.trange(total_length // STORE_BATCH_SIZE):
    data_items = []
    for jdx in range(STORE_BATCH_SIZE):
        real_jdx = batch_idx * STORE_BATCH_SIZE + jdx
        data_items.append(pt_db_list[real_jdx])
    maybe_get_video_frames(data_items=data_items, n_segments=N_SEGMENTS)
t = time.time() - s
print(t)