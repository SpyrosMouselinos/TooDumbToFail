import copy
import json
import os
import pickle
from typing import List, Dict, Any
import torch
import torch.nn as nn
import tqdm

from utils import load_db_json, get_video_frames, get_audio_frames, get_ocr_frames, OCR_RELEVANT_VIDEO_IDS_PATH, \
    gather_ocr_videos
from transformers import VideoMAEForVideoClassification, \
    VideoMAEImageProcessor, AutoFeatureExtractor, HubertModel
from models.OCR_model import OCRmodel



class PerceptionDataset:
    """Dataset class to store video items from dataset.

    Attributes:
      video_folder: Path to the folder containing the video / audio files.
      task: Task type for annotations.
      split: Dataset split to load.
      pt_db_list: List containing annotations for dataset according to
        split and task availability.
    """

    def __init__(self, pt_db_dict: Dict[str, Any],
                 video_folder: str,
                 task: str,
                 split: str,
                 js_only: bool,
                 use_audio: bool = False,
                 use_ocr: bool = False,
                 auto_unload: bool = True) -> None:
        """Initializes the PerceptionDataset class.

        Args:
          pt_db_dict (Dict): Dictionary containing annotations for dataset.
          video_folder (str): Path to the folder containing the videos.
          task (str): Task type for annotations.
          split (str): Dataset split to load.
          js_only (bool): Whether we load the full dataset or only the json-related description
          use_audio (bool): Whether we load the audio of each video segment
          use_ocr (bool): Whether we load the ocr module and apply it on every image (frames got by video segmentation default:16)
          auto_unload (bool): Whether we unload the video/audio encoder after every use (saves space, makes everything super slow)
        """
        self.js_only = js_only
        self.video_folder = video_folder
        self.task = task
        self.split = split
        self.pt_db_list = self.load_dataset(pt_db_dict)
        self.video_load_flag = False
        self.audio_load_flag = False
        self.ocr_load_flag = False
        self.use_audio = use_audio
        self.use_ocr = use_ocr
        self.auto_unload = auto_unload
        """
        Add conditional check to skip this if the video at hand is irrelevant to OCR
        """
        if os.path.exists(OCR_RELEVANT_VIDEO_IDS_PATH):
            with open(OCR_RELEVANT_VIDEO_IDS_PATH, 'r') as fin:
                self.ocr_relevance = json.load(fin)
        else:
            gather_ocr_videos(
                ['./data/mc_question_train.json', './data/mc_question_valid.json', './data/mc_question_test.json'])
            with open(OCR_RELEVANT_VIDEO_IDS_PATH, 'r') as fin:
                self.ocr_relevance = json.load(fin)

    def __unload_on_demand__(self, modality):
        if modality == 'video':
            self.video_processor = None
            self.video_model = None
            torch.cuda.empty_cache()
            self.video_load_flag = False
        elif modality == 'audio':
            self.audio_processor = None
            self.audio_model = None
            torch.cuda.empty_cache()
            self.audio_load_flag = False
        elif modality == 'ocr':
            self.ocr_processor = None
            self.ocr_model = None
            torch.cuda.empty_cache()
            self.ocr_load_flag = False
        else:
            raise ValueError(f'Argument modality should be video/audio/ocr... you gave {modality}\n')

    def __load_on_demand__(self, modality):
        if modality == 'video':
            self.video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            self.video_model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics")
            self.video_model.classifier = nn.Identity()
            self.video_model = self.video_model.to('cuda').half()
            self.video_load_flag = True
        elif modality == 'audio':
            self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.audio_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
            self.audio_model = self.audio_model.to('cuda')
            self.audio_load_flag = True
        elif modality == 'ocr':
            self.ocr_processor = lambda x: x
            self.ocr_model = OCRmodel(use_angle_cls=True, det_db_thresh=0.1, label_list=['0'], dual_pass=True,
                                      crop_scale=1.5)
            self.ocr_load_flag = True
        else:
            raise ValueError(f'Argument modality should be video/audio... you gave {modality}\n')

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
        if not self.video_load_flag:
            self.__load_on_demand__('video')
        N_SEGMENTS = 1
        BATCH_SIZE = 16
        loaded_videos = []
        vid_frames = get_video_frames(data_item, self.video_folder, override_video_name=False, resize_to=224,
                                      num_samples=16, n_segments=N_SEGMENTS)
        for s in range(N_SEGMENTS):
            for f in vid_frames[s]:
                loaded_videos.append(f)
        inputs = self.video_processor(loaded_videos, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].resize(N_SEGMENTS, 16, inputs['pixel_values'].size()[2],
                                                               inputs['pixel_values'].size()[3],
                                                               inputs['pixel_values'].size()[4])
        final_logits = []
        if N_SEGMENTS >= BATCH_SIZE:
            for i in range(N_SEGMENTS // BATCH_SIZE):
                semi_input = inputs['pixel_values'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE].to('cuda')
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    with torch.no_grad():
                        outputs = self.video_model(pixel_values=semi_input)
                        logits = outputs.logits
                        for f in logits:
                            final_logits.append(f)
        else:
            semi_input = inputs['pixel_values'].to('cuda')
            with torch.cuda.amp.autocast(dtype=torch.float16):
                with torch.no_grad():
                    outputs = self.video_model(pixel_values=semi_input)
                    logits = outputs.logits
                    for f in logits:
                        final_logits.append(f)
        vid_frames = torch.stack(final_logits, dim=0).mean(0)
        if self.auto_unload:
            self.__unload_on_demand__('video')
        return vid_frames

    def _get_audio_frames(self, data_item):
        if not self.audio_load_flag:
            self.__load_on_demand__('audio')
        N_SEGMENTS = 1
        BATCH_SIZE = 16
        loaded_audios = []
        aud_frames = get_audio_frames(data_item, self.video_folder, override_video_name=False, num_samples=None,
                                      n_segments=1)
        for s in range(N_SEGMENTS):
            loaded_audios.append(aud_frames[s])
        inputs = self.audio_processor(loaded_audios, sampling_rate=16000, return_tensors="pt", padding=True)
        final_logits = []
        if N_SEGMENTS >= BATCH_SIZE:
            for i in range(N_SEGMENTS // BATCH_SIZE):
                semi_input = inputs['input_values'][i * BATCH_SIZE: (i + 1) * BATCH_SIZE].to('cuda')
                with torch.no_grad():
                    outputs = self.audio_model(input_values=semi_input)
                    logits = outputs['last_hidden_state'].mean(1)
                    for f in logits:
                        final_logits.append(f)
        else:
            semi_input = inputs['input_values'].to('cuda')
            with torch.no_grad():
                outputs = self.audio_model(input_values=semi_input)
                logits = outputs['last_hidden_state'].mean(1)
                for f in logits:
                    final_logits.append(f)

        aud_frames = torch.stack(final_logits, dim=0).mean(0)
        if self.auto_unload:
            self.__unload_on_demand__('audio')
        return aud_frames

    def _get_ocr_frames(self, data_item):
        if not self.ocr_load_flag:
            self.__load_on_demand__('ocr')
        N_SEGMENTS = 1
        if data_item['metadata']['video_id'] in self.ocr_relevance:
            vid_frames = get_ocr_frames(data_item, self.video_folder, override_video_name=False, resize_to=300,
                                        num_samples=16, n_segments=N_SEGMENTS)[0]
            ocr_frames = []
            inputs = self.ocr_processor(vid_frames)
            identified_strings = self.ocr_model(inputs)
            for f in identified_strings:
                ocr_frames.append(f)
        else:
            ocr_frames = []
        # if self.auto_unload:
        #     self.__unload_on_demand__('ocr') No need to unload OCR module (It is superlightweight)
        return ocr_frames

    def _maybe_get_video_frames(self, data_item, n_segments=1):
        if isinstance(self.video_folder, str):
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
        elif isinstance(self.video_folder, tuple):
            for vf in self.video_folder:
                try:
                    video_file_pickle = os.path.join(vf,
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
                except:
                    pass
        return vid_frames

    def _maybe_get_audio_frames(self, data_item, n_segments=1):
        if isinstance(self.video_folder, str):
            audio_file_pickle = os.path.join(self.video_folder,
                                             data_item['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
            if os.path.exists(audio_file_pickle):
                # Unpickle and return
                with open(audio_file_pickle, 'rb') as fin:
                    aud_frames = pickle.load(fin)
            else:
                # Get video frame and store
                aud_frames = self._get_audio_frames(data_item=data_item)
                with open(audio_file_pickle, 'wb') as fout:
                    pickle.dump(aud_frames, fout)
        elif isinstance(self.video_folder, tuple):
            for vf in self.video_folder:
                try:
                    audio_file_pickle = os.path.join(vf,
                                                     data_item['metadata']['video_id']) + f'_audioseg_{n_segments}.pkl'
                    if os.path.exists(audio_file_pickle):
                        # Unpickle and return
                        with open(audio_file_pickle, 'rb') as fin:
                            aud_frames = pickle.load(fin)
                    else:
                        # Get video frame and store
                        aud_frames = self._get_audio_frames(data_item=data_item)
                        with open(audio_file_pickle, 'wb') as fout:
                            pickle.dump(aud_frames, fout)
                except:
                    pass
        return aud_frames

    def _maybe_get_ocr_frames(self, data_item, n_segments=1):
        if isinstance(self.video_folder, str):
            video_file_pickle = os.path.join(self.video_folder,
                                             data_item['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
            if os.path.exists(video_file_pickle):
                # Unpickle and return
                with open(video_file_pickle, 'rb') as fin:
                    ocr_frames = pickle.load(fin)['ocr']
            else:
                # Get video frame and store
                ocr_frames = self._get_ocr_frames(data_item=data_item)
                with open(video_file_pickle, 'wb') as fout:
                    pickle.dump({'ocr': ocr_frames}, fout)
        elif isinstance(self.video_folder, tuple):
            for vf in self.video_folder:
                try:
                    video_file_pickle = os.path.join(vf,
                                                     data_item['metadata']['video_id']) + f'_ocrseg_{n_segments}.pkl'
                    if os.path.exists(video_file_pickle):
                        # Unpickle and return
                        with open(video_file_pickle, 'rb') as fin:
                            ocr_frames = pickle.load(fin)['ocr']
                    else:
                        # Get video frame and store
                        ocr_frames = self._get_ocr_frames(data_item=data_item)
                        with open(video_file_pickle, 'wb') as fout:
                            pickle.dump({'ocr': ocr_frames}, fout)
                except:
                    pass
        return ocr_frames

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
            if self.use_audio:
                aud_frames = torch.rand(size=(1, 768))
            else:
                aud_frames = None
            if self.use_ocr:
                ocr_frames = []
            else:
                ocr_frames = None
        else:
            vid_frames = self._maybe_get_video_frames(data_item=data_item, n_segments=1)
            if self.use_audio:
                aud_frames = self._maybe_get_audio_frames(data_item=data_item, n_segments=1)
            else:
                aud_frames = None
            if self.use_ocr:
                ocr_frames = self._maybe_get_ocr_frames(data_item=data_item, n_segments=1)
            else:
                ocr_frames = None

        return {'metadata': metadata,
                self.task: annot,
                'frames': vid_frames,
                'audio': aud_frames,
                'ocr': ocr_frames
                }

    def union(self, otherDataset):
        # Check for same arguments #
        assert self.js_only == otherDataset.js_only
        assert self.task == otherDataset.task
        assert self.video_load_flag == otherDataset.video_load_flag
        assert self.audio_load_flag == otherDataset.audio_load_flag
        assert self.ocr_load_flag == otherDataset.ocr_load_flag
        assert self.use_audio == otherDataset.use_audio
        assert self.use_ocr == otherDataset.use_ocr
        assert self.auto_unload == otherDataset.auto_unload

        # If everything is ok you need to merge the pt_db_list and update the length of the new dataset
        print(f"Old dataset length: {len(self.pt_db_list)}")
        self.pt_db_list += otherDataset.pt_db_list
        print(f"New dataset length: {len(self.pt_db_list)}")
        print("Setting mixed video folder...")
        self.video_folder = (self.video_folder, otherDataset.video_folder)
        return

    def __add__(self, other):
        self.union(other)
        return self


def atest_dataset_with_json_only():
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


def atest_dataset_with_video():
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


def atest_dataset_with_video_and_audio():
    cfg = {'video_folder': './data/train/videos/',
           'task': 'mc_question',
           'split': 'train',
           'js_only': False,
           'use_audio': True,
           }

    train_db_path = 'data/mc_question_train.json'
    train_db_dict = load_db_json(train_db_path)
    dataset = PerceptionDataset(train_db_dict, **cfg)
    for item in tqdm.tqdm(dataset):
        pass
    print("End")


def atest_dataset_union():
    train_db_path = 'data/mc_question_train.json'
    train_db_dict = load_db_json(train_db_path)

    valid_db_path = 'data/mc_question_valid.json'
    valid_db_dict = load_db_json(valid_db_path)

    train_cfg = {'video_folder': './data/train/videos/',
                 'task': 'mc_question',
                 'split': 'train',
                 'js_only': False,
                 'use_audio': True,
                 }
    train_mc_vqa_dataset = PerceptionDataset(train_db_dict, **train_cfg)
    valid_cfg = {'video_folder': './data/valid/',
                 'task': 'mc_question',
                 'split': 'valid',
                 'js_only': False,
                 'use_audio': True,
                 }
    val_mc_vqa_dataset = PerceptionDataset(valid_db_dict, **valid_cfg)

    original_length = copy.deepcopy(len(train_mc_vqa_dataset))

    train_mc_vqa_dataset.union(val_mc_vqa_dataset)
    for i, item in enumerate(train_mc_vqa_dataset):
        if i < original_length - 1:
            pass
        else:
            print(item)

            break

    for i in val_mc_vqa_dataset:
        print(i - item)
        break

    print("End")


def atest_dataset_with_video_and_audio_and_ocr():
    cfg = {'video_folder': './data/test/',
           'task': 'mc_question',
           'split': 'test',
           'js_only': False,
           'use_audio': True,
           'use_ocr': True,
           }

    train_db_path = 'data/mc_question_test.json'
    train_db_dict = load_db_json(train_db_path)
    dataset = PerceptionDataset(train_db_dict, **cfg)
    for item in tqdm.tqdm(dataset):
        pass
    print("End")


if __name__ == '__main__':
    # test_dataset_with_json_only()
    # test_dataset_with_video_and_audio()
    # atest_dataset_union()
    # test_dataset_addition()
    atest_dataset_with_video_and_audio_and_ocr()
