# """
###
# @title Serialize Example Results File
# Writing model outputs in the expected competition format. This
# JSON file contains answers to all questions in the validation split in the
# format:
#
# example_output = {
# 	'video_1009' :{
# 		'mc_question':[
# 			{'id': 0, 'answer_id': 2, 'answer': 'static or shaking'}
# 		]
# 	}
# }
#
# This file could be used directly as a submission for the Eval.ai challenge

# with open('results.json', 'w') as my_file:
#     json.dump(answers, my_file)
# """
import json
import os
import random
import zipfile

import av
import cv2
import moviepy.editor as mvp
import numpy as np
import requests
from typing import Dict, Any
import decord as de

ctx = de.cpu(0)

# from data import PerceptionDataset
from models.FreqBaseline import FreqMCVQABaseline

AREA = ['physics', 'semantics', 'abstraction', 'memory']
REASONING = ['descriptive', 'counterfactual', 'explanatory', 'predictive']
TAG = ['motion', 'place recognition', 'action counting', 'spatial relations',
       'object recognition', 'action recognition', 'distractor action',
       'distractor object', 'state recognition', 'object counting',
       'change detection', 'sequencing', 'task completion',
       'adversarial action', 'collision', 'object attributes',
       'pattern breaking', 'feature matching', 'pattern discovery',
       'part recognition', 'material', 'event recall', 'containment',
       'occlusion', 'solidity', 'object permanence', 'colour recognition',
       'conservation', 'quantity', 'event counting', 'language',
       'visual discrimination', 'general knowledge', 'stability']

TAG_AREA = {
    'change detection': 'memory',
    'event recall': 'memory',
    'sequencing': 'memory',
    'visual discrimination': 'memory',
    'action counting': 'abstraction',
    'event counting': 'abstraction',
    'feature matching': 'abstraction',
    'object counting': 'abstraction',
    'pattern breaking': 'abstraction',
    'pattern discovery': 'abstraction',
    'collision': 'physics',
    'conservation': 'physics',
    'containment': 'physics',
    'material': 'physics',
    'motion': 'physics',
    'occlusion': 'physics',
    'object attributes': 'physics',
    'object permanence': 'physics',
    'quantity': 'physics',
    'solidity': 'physics',
    'spatial relations': 'physics',
    'stability': 'physics',
    'action recognition': 'semantics',
    'adversarial action': 'semantics',
    'colour recognition': 'semantics',
    'distractor action': 'semantics',
    'distractor object': 'semantics',
    'general knowledge': 'semantics',
    'language': 'semantics',
    'object recognition': 'semantics',
    'part recognition': 'semantics',
    'place recognition': 'semantics',
    'state recognition': 'semantics',
    'task completion': 'semantics',
}

CAT = AREA + REASONING + TAG
DATA_PATH = data_path = './data/'
TRAIN_DATA_PATH = train_data_path = DATA_PATH + 'train/'
VALID_DATA_PATH = valid_data_path = DATA_PATH + 'valid/'


def download_and_unzip(url: str, destination: str):
    """Downloads and unzips a .zip file to a destination.

    Downloads a file from the specified URL, saves it to the destination
    directory, and then extracts its contents.

    If the file is larger than 1GB, it will be downloaded in chunks,
    and the download progress will be displayed.

    Args:
      url (str): The URL of the file to download.
      destination (str): The destination directory to save the file and
        extract its contents.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)

    filename = url.split('/')[-1]
    file_path = os.path.join(destination, filename)

    if os.path.exists(file_path):
        print(f'{filename} already exists. Skipping download.')
        return

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    gb = 1024 * 1024 * 1024

    if total_size / gb > 1:
        print(f'{filename} is larger than 1GB, downloading in chunks')
        chunk_flag = True
        chunk_size = int(total_size / 100)
    else:
        chunk_flag = False
        chunk_size = total_size

    with open(file_path, 'wb') as file:
        for chunk_idx, chunk in enumerate(
                response.iter_content(chunk_size=chunk_size)):
            if chunk:
                if chunk_flag:
                    print(f"""{chunk_idx}% downloading
          {round((chunk_idx * chunk_size) / gb, 1)}GB
          / {round(total_size / gb, 1)}GB""")
                file.write(chunk)
    print(f"'{filename}' downloaded successfully.")

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination)
    print(f"'{filename}' extracted successfully.")

    os.remove(file_path)


def load_db_json(db_file: str) -> Dict[str, Any]:
    """Loads a JSON file as a dictionary.

    Args:
      db_file (str): Path to the JSON file.

    Returns:
      Dict: Loaded JSON data as a dictionary.

    Raises:
      FileNotFoundError: If the specified file doesn't exist.
      TypeError: If the JSON file is not formatted as a dictionary.
    """
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f'No such file: {db_file}')

    with open(db_file, 'r') as f:
        db_file_dict = json.load(f)
        if not isinstance(db_file_dict, dict):
            raise TypeError('JSON file is not formatted as a dictionary.')
        return db_file_dict


def load_mp4_to_frames(filename: str, indices=None, resize_to=None) -> np.array:
    """Loads an MP4 video file and returns its frames as a NumPy array.

    Args:
      filename (str): Path to the MP4 video file.

    Returns:
      np.array: Frames of the video as a NumPy array.
    """

    vr = de.VideoReader(filename, ctx=ctx, width=resize_to, height=resize_to)
    if indices is None:
        indices = list(range(len(vr)))
    frames = vr.get_batch(indices).asnumpy()
    return frames


def get_video_frames(data_item: Dict[str, Any],
                     video_folder_path: str,
                     override_video_name: bool = False, resize_to=None, num_samples=16, n_segments=1) -> np.array:
    """Loads frames of a video specified by an item dictionary.

    Assumes format of annotations used in the Perception Test Dataset.

    Args:
      data_item (Dict): Item from dataset containing metadata.
      video_folder_path (str): Path to the directory containing videos.

    Returns:
      np.array: Frames of the video as a NumPy array.
      :param override_video_name: Add one random video from the pre-existing ones
    """

    if override_video_name:
        video_file = os.path.join(video_folder_path,
                                  'video_1580') + '.mp4'
    else:
        video_file = os.path.join(video_folder_path,
                                  data_item['metadata']['video_id']) + '.mp4'

    parts = []
    t = data_item['metadata']['num_frames']
    assert num_samples > 0 and t > 0
    vr = de.VideoReader(video_file, ctx=ctx, width=resize_to, height=resize_to)
    for i in range(n_segments):
        indices = np.linspace(i * (t // n_segments), (i + 1) * ((t // n_segments) - 2), num_samples)
        indices = np.clip(indices, i * (t // n_segments), (i + 1) * ((t // n_segments) - 2)).astype(int)
        parts.append(vr.get_batch(indices).asnumpy())
    return parts


# def video_temporal_subsample(
#         x: np.ndarray, num_samples: int, temporal_dim: int = 0, n_segments: int = 1):
#     """
#     Uniformly subsamples num_samples indices from the temporal dimension of the video.
#     Returns:
#         An x-like with subsampled temporal dimension.
#     """
#     t = x.shape[temporal_dim]
#     assert num_samples > 0 and t > 0
#     # Sample by nearest neighbor interpolation if num_samples > t.
#     parts = []
#     for i in range(n_segments):
#         indices = np.linspace(i * (t // n_segments), (i + 1) * ((t // n_segments) - 1), num_samples)
#         indices = np.clip(indices, i * (t // n_segments), (i + 1) * ((t // n_segments) - 1)).astype(int)
#         for f in np.take(x, indices, axis=temporal_dim).astype('uint8'):
#             parts.append(f)
#     return parts


def test_video(valid_db_dict, video_id='video_8241'):
    video_path = 'data/sample/videos/'
    video_item = valid_db_dict[video_id]
    video_path = os.path.join(video_path,
                              video_item['metadata']['video_id']) + '.mp4'
    mvp.ipython_display(video_path)

    print('Video ID: ', video_item['metadata']['video_id'])
    print('Video Length: ', video_item['metadata']['num_frames'])
    print('Video FPS: ', video_item['metadata']['frame_rate'])
    print('Video  Resolution: ', video_item['metadata']['resolution'])

    if video_item['mc_question']:
        for example_q in video_item['mc_question']:
            print('---------------------------------')
            print('Question: ', example_q['question'])
            print('Options: ', example_q['options'])
            print('Answer ID: ', example_q['answer_id'])
            print('Answer: ', example_q['options'][example_q['answer_id']])
            print('Question info: ')
            print('Reasoning: ', example_q['reasoning'])
            print('Tag: ', example_q['tag'])
            print('area: ', example_q['area'])
            print('---------------------------------')


def calc_top1(answers_dict: Dict[str, Any],
              db_dict: Dict[str, Any]) -> float:
    """Calculates the top-1 accuracy and results for each category.

    Args:
      answers_dict (Dict): Dictionary containing the answers.
      db_dict (Dict): Dictionary containing the database.

    Returns:
      float: Top-1 accuracy.

    Raises:
      KeyError: Raises error if the results contain a question ID that does not
        exist in the annotations.
      ValueError: Raises error if the answer ID is outside the expected range
        [0,2].
      ValueError: Raises error if the text of the answer in the results does not
        match the expected string answer in the annotations.
      ValueError: If answers are missing from the results.

    """
    expected_total = 0
    total_correct = 0
    total = 0

    for v in db_dict.values():
        expected_total += len(v['mc_question'])

    for vid_id, vid_answers in answers_dict.items():
        for answer_info in vid_answers:
            answer_id = answer_info['answer_id']
            question_idx = answer_info['id']

            try:
                ground_truth = db_dict[vid_id]['mc_question'][question_idx]
            except KeyError as exc:
                print(f'Unexpected question ID in {vid_id}.')
                continue

            if answer_id > 2 or answer_id < 0:
                raise ValueError(f'Answer ID must be in range [0:2], got {answer_id}.')
            if ground_truth['options'][answer_id] != answer_info['answer']:
                raise ValueError('Answer text is not as expected.')

            gt_answer_id = ground_truth['answer_id']
            val = int(gt_answer_id == answer_id)
            total_correct += val
            total += 1

    if expected_total != total:
        raise ValueError('Missing answers in results.')

    return total_correct / total


def calc_top1_by_cat(answers_dict: Dict[str, Any],
                     db_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates the top-1 accuracy and results for each category.

    Args:
      answers_dict (Dict): Dictionary containing the answers.
      db_dict (Dict): Dictionary containing the database.

    Returns:
      Dict: Top-1 accuracy and results for each category.
    """
    results_dict = {k: v for k, v in zip(CAT, np.zeros((len(CAT), 2)))}

    for vid_id, vid_answers in answers_dict.items():
        for answer_info in vid_answers:
            answer_id = answer_info['answer_id']
            question_idx = answer_info['id']
            ground_truth = db_dict[vid_id]['mc_question'][question_idx]
            gt_answer_id = ground_truth['answer_id']
            val = int(gt_answer_id == answer_id)

            used_q_areas = []
            q_areas = [TAG_AREA[tag] for tag in ground_truth['tag']]
            for a in q_areas:
                if a not in used_q_areas:
                    results_dict[a][0] += val
                    results_dict[a][1] += 1
                    used_q_areas.append(a)

            results_dict[ground_truth['reasoning']][0] += val
            results_dict[ground_truth['reasoning']][1] += 1
            for t in ground_truth['tag']:
                results_dict[t][0] += val
                results_dict[t][1] += 1

    results_dict = {k: v[0] / v[1] for k, v in results_dict.items()}
    return results_dict


def analyse_test_results(results, category='8'):
    shot_results = results[category]
    print('8-shot Frequency Results')
    print('---------------------------------')

    print('Areas')
    print('---------------------------------')
    for area in AREA:
        print(f'{area}: {shot_results[area] * 100:.2f}%')
    print('---------------------------------')

    print('Reasonings')
    print('---------------------------------')
    for reasoning in REASONING:
        print(f'{reasoning}: {shot_results[reasoning] * 100:.2f}%')
    print('---------------------------------')

    print('Tags')
    print('---------------------------------')
    for example_tag in TAG:
        print(f'{example_tag}: {shot_results[example_tag] * 100:.2f}%')


def test_download_samples():
    sample_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip'
    download_and_unzip(sample_annot_url, data_path)
    sample_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip'
    download_and_unzip(sample_videos_url, data_path)
    sample_audios_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sample_audios.zip'
    download_and_unzip(sample_audios_url, data_path)

    # validation set annotations to perform tracking
    valid_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip'
    download_and_unzip(valid_annot_url, data_path)
    # validation set annotations to perform tracking
    train_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip'
    download_and_unzip(train_annot_url, data_path)

    # validation videos not downloaded because they are too big (approx 70GB).
    # not needed for this baseline since we are choosing answers based on the
    # frequency of oocurence we do not actually need the videos to calculate
    # the performance.

    train_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/train_videos.zip'
    download_and_unzip(train_videos_url, data_path + '/train')
    # valid_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/valid_videos.zip'
    # download_and_unzip(valid_videos_url, data_path)
