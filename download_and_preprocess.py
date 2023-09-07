import json
import os
import pickle
import time
import torch
import tqdm
import numpy as np
from utils import load_db_json, get_video_frames, get_audio_frames, get_ocr_frames, OCR_RELEVANT_VIDEO_IDS_PATH, \
    gather_ocr_videos, get_qo_frames
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from transformers import AutoFeatureExtractor, HubertModel
from models.OCR_model import OCRmodel

# Constants

EXTRACTED_MODALITY = 'Ocr'
SPLIT = split = 'train'
train_db_path = f'data/mc_question_{SPLIT}.json'
pt_db_dict = load_db_json(train_db_path)

video_folder = f'./data/{SPLIT}/' if SPLIT != 'train' else f'./data/{SPLIT}/videos'
task = 'mc_question'
N_SEGMENTS = 1
MINI_BATCH_SIZE = 16
STORE_BATCH_SIZE = 4

if os.path.exists(OCR_RELEVANT_VIDEO_IDS_PATH):
    with open(OCR_RELEVANT_VIDEO_IDS_PATH, 'r') as fin:
        OCR_RELEVANCE = json.load(fin)
else:
    gather_ocr_videos(
        ['./data/mc_question_train.json', './data/mc_question_valid.json', './data/mc_question_test.json'])
    with open(OCR_RELEVANT_VIDEO_IDS_PATH, 'r') as fin:
        OCR_RELEVANCE = json.load(fin)


def load_dataset(d):
    l = []
    for _, v in d.items():
        if v['metadata']['split'] == split:
            if v[task]:
                l.append(v)
    return l


pt_db_list = load_dataset(pt_db_dict)
total_length = len(pt_db_list)
if EXTRACTED_MODALITY == 'Video':
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model.classifier = nn.Identity()
    model = model.to('cuda').half()
elif EXTRACTED_MODALITY == 'Audio':
    processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to('cuda')
elif EXTRACTED_MODALITY == 'Ocr':
    model = OCRmodel(dual_pass=True)
    model.to('cuda')
elif EXTRACTED_MODALITY == 'QO':
    processor = lambda x: torch.from_numpy(x).float()
    model = lambda x: x
else:
    raise NotImplementedError


def get_video_frames_wrapper(data_items):
    loaded_videos = []
    for i in range(STORE_BATCH_SIZE):
        vid_frames = get_video_frames(data_items[i], video_folder, override_video_name=False, resize_to=224,
                                      num_samples=16, n_segments=N_SEGMENTS)
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


def get_ocr_frames_wrapper(data_items):
    loaded_ocrs = []
    for i in range(STORE_BATCH_SIZE):
        vid_frames = get_ocr_frames(data_items[i],
                                    video_folder,
                                    override_video_name=False,
                                    resize_to=300,
                                    num_samples=16,
                                    n_segments=N_SEGMENTS)
        for s in range(N_SEGMENTS):
            loaded_ocrs.append(vid_frames[s])
    ocr_frames = []
    loaded_ocrs = np.concatenate(loaded_ocrs, axis=0)
    for i in range(len(loaded_ocrs) // MINI_BATCH_SIZE):
        slice = loaded_ocrs[MINI_BATCH_SIZE * i: MINI_BATCH_SIZE * (i + 1)]
        ocr_frames.append(model(slice))
    assert len(ocr_frames) == STORE_BATCH_SIZE
    return ocr_frames


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


def maybe_get_ocr_frames(data_items, n_segments=1):
    for i in range(len(data_items)):
        video_file_pickle = os.path.join(video_folder,
                                         data_items[i]['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
        if os.path.exists(video_file_pickle) or not (data_items[i]['metadata']['video_id'] in OCR_RELEVANCE):
            continue
        else:
            ocr_frames = get_ocr_frames_wrapper(data_items=data_items)
            for j in range(len(data_items)):
                video_file_pickle = os.path.join(video_folder,
                                                 data_items[j]['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
                with open(video_file_pickle, 'wb') as fout:
                    pickle.dump({'ocr': ocr_frames[j]}, fout)
            return
    return


def get_audio_frames_wrapper(data_items):
    loaded_audios = []
    for i in range(STORE_BATCH_SIZE):
        aud_frames = get_audio_frames(data_items[i],
                                      video_folder,
                                      override_video_name=False,
                                      num_samples=None,
                                      n_segments=None)
        for s in range(N_SEGMENTS):
            loaded_audios.append(aud_frames[s])

    inputs = processor(loaded_audios, sampling_rate=16000, return_tensors="pt", padding=True)
    final_logits = []
    if N_SEGMENTS >= MINI_BATCH_SIZE:
        for i in range(N_SEGMENTS // MINI_BATCH_SIZE):
            semi_input = inputs['input_values'][i * MINI_BATCH_SIZE: (i + 1) * MINI_BATCH_SIZE].to('cuda')
            with torch.no_grad():
                outputs = model(input_values=semi_input)
                final_logits = outputs['last_hidden_state'].mean(1)
    else:
        semi_input = inputs['input_values'].to('cuda')
        with torch.no_grad():
            outputs = model(input_values=semi_input)
            final_logits = outputs['last_hidden_state'].mean(1)
    aud_frames = final_logits
    return aud_frames


def maybe_get_audio_frames(data_items, n_segments=1):
    for i in range(len(data_items)):
        audio_file_pickle = os.path.join(video_folder,
                                         data_items[i]['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
        if os.path.exists(audio_file_pickle):
            continue
        else:
            aud_frames = get_audio_frames_wrapper(data_items=data_items)
            for j in range(len(data_items)):
                audio_file_pickle = os.path.join(video_folder,
                                                 data_items[j]['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
                with open(audio_file_pickle, 'wb') as fout:
                    pickle.dump(aud_frames[j], fout)
            return
    return


def get_qo_frames_wrapper(data_items):
    loaded_questions = []
    loaded_options = []
    for i in range(STORE_BATCH_SIZE):
        q_frames, o_frames = get_qo_frames(data_items[i],
                                           video_folder,
                                           override_video_name=False,
                                           num_samples=None,
                                           n_segments=None)
        for s in range(N_SEGMENTS):
            loaded_questions.append(q_frames[s])
            loaded_options.append(o_frames[s])
    q_frames = []
    o_frames = []
    loaded_q = np.concatenate(loaded_questions, axis=0)
    loaded_o = np.concatenate(loaded_options, axis=0)
    for i in range(len(loaded_q) // MINI_BATCH_SIZE):
        slice = loaded_q[MINI_BATCH_SIZE * i: MINI_BATCH_SIZE * (i + 1)]
        q_frames.append(model(slice))
        slice = loaded_o[MINI_BATCH_SIZE * i: MINI_BATCH_SIZE * (i + 1)]
        o_frames.append(model(slice))
    assert len(q_frames) == STORE_BATCH_SIZE
    assert len(o_frames) == STORE_BATCH_SIZE
    return q_frames, o_frames


def maybe_get_qo_frames(data_items, n_segments=1):
    for i in range(len(data_items)):
        audio_file_pickle = os.path.join(video_folder,
                                         data_items[i]['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
        if os.path.exists(audio_file_pickle):
            continue
        else:
            aud_frames = get_audio_frames_wrapper(data_items=data_items)
            for j in range(len(data_items)):
                audio_file_pickle = os.path.join(video_folder,
                                                 data_items[j]['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
                with open(audio_file_pickle, 'wb') as fout:
                    pickle.dump(aud_frames[j], fout)
            return
    return


def maybe_get_ocr_frames(data_items, n_segments=1):
    for i in range(len(data_items)):
        video_file_pickle = os.path.join(video_folder,
                                         data_items[i]['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
        if os.path.exists(video_file_pickle) or not (data_items[i]['metadata']['video_id'] in OCR_RELEVANCE):
            continue
        else:
            ocr_frames = get_ocr_frames_wrapper(data_items=data_items)
            for j in range(len(data_items)):
                video_file_pickle = os.path.join(video_folder,
                                                 data_items[j]['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
                with open(video_file_pickle, 'wb') as fout:
                    pickle.dump({'ocr': ocr_frames[j]}, fout)
            return
    return


if EXTRACTED_MODALITY == 'Video':
    s = time.time()
    for batch_idx in tqdm.trange(total_length // STORE_BATCH_SIZE):
        data_items = []
        for jdx in range(STORE_BATCH_SIZE):
            real_jdx = batch_idx * STORE_BATCH_SIZE + jdx
            data_items.append(pt_db_list[real_jdx])
        maybe_get_video_frames(data_items=data_items, n_segments=N_SEGMENTS)
    t = time.time() - s
    print(t)
elif EXTRACTED_MODALITY == 'Audio':
    s = time.time()
    for batch_idx in tqdm.trange(total_length // STORE_BATCH_SIZE):
        data_items = []
        for jdx in range(STORE_BATCH_SIZE):
            real_jdx = batch_idx * STORE_BATCH_SIZE + jdx
            data_items.append(pt_db_list[real_jdx])
        maybe_get_audio_frames(data_items=data_items, n_segments=1)
    t = time.time() - s
    print(t)
elif EXTRACTED_MODALITY == 'Ocr':
    s = time.time()
    for batch_idx in tqdm.trange(total_length // STORE_BATCH_SIZE):
        data_items = []
        for jdx in range(STORE_BATCH_SIZE):
            real_jdx = batch_idx * STORE_BATCH_SIZE + jdx
            data_items.append(pt_db_list[real_jdx])
        maybe_get_ocr_frames(data_items=data_items, n_segments=1)
    t = time.time() - s
    print(t)
else:
    raise NotImplementedError
