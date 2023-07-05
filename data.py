import os
import pickle
from typing import List, Dict, Any

import tqdm

from utils import load_db_json, get_video_frames
from transformers import AutoImageProcessor, TimesformerForVideoClassification, VideoMAEForVideoClassification, \
    VideoMAEImageProcessor
import numpy as np
import torch
import torch.nn as nn


class PerceptionDataset:
    """Dataset class to store video items from dataset.

    Attributes:
      video_folder: Path to the folder containing the videos.
      task: Task type for annotations.
      split: Dataset split to load.
      pt_db_list: List containing annotations for dataset according to
        split and task availability.
    """

    def __init__(self, pt_db_dict: Dict[str, Any], video_folder: str,
                 task: str, split: str, js_only: bool) -> None:
        """Initializes the PerceptionDataset class.

        Args:
          pt_db_dict (Dict): Dictionary containing annotations for dataset.
          video_folder (str): Path to the folder containing the videos.
          task (str): Task type for annotations.
          split (str): Dataset split to load.
          js_only (bool): Whether we load the full dataset or only the json-related description
        """
        self.js_only = js_only
        self.video_folder = video_folder
        self.task = task
        self.split = split
        self.pt_db_list = self.load_dataset(pt_db_dict)
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.model.classifier = nn.Identity()
        self.model = self.model.to('cuda').half()

    def load_dataset(self, pt_db_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Loads the dataset from the annotation file and processes.

        Dict is processed according to split and task.

        Args:
          pt_db_dict: (Dict): Dictionary containing
            annotations.

        Returns:
          List: List of database items containing annotations.
        """
        pt_db_list = []
        for _, v in pt_db_dict.items():
            if v['metadata']['split'] == self.split:
                if v[self.task]:  # If video has annotations for this task
                    pt_db_list.append(v)

        return pt_db_list

    def __len__(self) -> int:
        """Returns the total number of videos in the dataset.

        Returns:
          int: Total number of videos.
        """
        return len(self.pt_db_list)

    def _get_video_frames(self, data_item):
        N_SEGMENTS = 1
        BATCH_SIZE = 16
        loaded_videos = []
        vid_frames = get_video_frames(data_item, self.video_folder, override_video_name=False, resize_to=224, num_samples=16, n_segments=N_SEGMENTS)
        for s in range(N_SEGMENTS):
            for f in vid_frames[s]:
                loaded_videos.append(f)
        inputs = self.processor(loaded_videos, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].resize(N_SEGMENTS, 16, inputs['pixel_values'].size()[2],
                                                               inputs['pixel_values'].size()[3],
                                                               inputs['pixel_values'].size()[4])
        final_logits = []
        if N_SEGMENTS >= BATCH_SIZE:
            for i in range(N_SEGMENTS // BATCH_SIZE):
                semi_input = inputs['pixel_values'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE].to('cuda')
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    with torch.no_grad():
                        outputs = self.model(pixel_values=semi_input)
                        logits = outputs.logits
                        for f in logits:
                            final_logits.append(f)
        else:
            semi_input = inputs['pixel_values'].to('cuda')
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    outputs = self.model(pixel_values=semi_input)
                    logits = outputs.logits
                    for f in logits:
                        final_logits.append(f)
        vid_frames = torch.stack(final_logits, dim=0).mean(0)
        return vid_frames

    def _maybe_get_video_frames(self, data_item, n_segments=1):
        video_file_pickle = os.path.join(self.video_folder,
                                         data_item['metadata']['video_id']) + f'_seg_{n_segments}.pkl'
        if os.path.exists(video_file_pickle):
            # Unpickle and return
            with open(video_file_pickle, 'rb') as fin:
                vid_frames = pickle.load(fin)
        else:
            # Get video frame and store
            vid_frames = self._get_video_frames(data_item=data_item)
            with open(video_file_pickle, 'wb') as fout:
                pickle.dump(vid_frames, fout)
        return vid_frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the video and annotations for a given index.

          example_annotation = {
            'video_10909':{
              'mc_question': [
                {'id': 0, 'question': 'Is the camera moving or static?',
                      'options': ["I don't know", 'moving', 'static or shaking'],
                      'answer_id': 2, 'area': 'physics', 'reasoning': 'descriptive',
                      'tag': ['motion']
                }
              ]
            }
          }

        Args:
          idx (int): Index of the video.

        Returns:
          Dict: Dictionary containing the video frames, metadata, annotations.
        """
        data_item = self.pt_db_list[idx]
        annot = data_item[self.task]
        metadata = data_item['metadata']
        # here we are loading a placeholder as the frames
        # the commented out function below will actually load frames
        if self.js_only:
            vid_frames = torch.rand(size=(1, 768))
        else:
            vid_frames = self._maybe_get_video_frames(data_item=data_item, n_segments=1)
        return {'metadata': metadata,
                self.task: annot,
                'frames': vid_frames}


def test_dataset_with_json_only():
    cfg = {'video_folder': './data/videos/',
           'task': 'mc_question',
           'split': 'valid',
           'js_only': True}

    valid_db_path = 'data/mc_question_valid.json'
    valid_db_dict = load_db_json(valid_db_path)
    dataset = PerceptionDataset(valid_db_dict, **cfg)
    for item in dataset:
        print(item)
    print("End")


def test_dataset_with_video():
    cfg = {'video_folder': './data/sample/videos/',
           'task': 'mc_question',
           'split': 'valid',
           'js_only': False}

    valid_db_path = 'data/mc_question_valid.json'
    valid_db_dict = load_db_json(valid_db_path)
    dataset = PerceptionDataset(valid_db_dict, **cfg)
    for item in tqdm.tqdm(dataset):
        pass
    print("End")


if __name__ == '__main__':
    # test_dataset_with_json_only()
    test_dataset_with_video()
