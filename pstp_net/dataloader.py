import pickle
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import ast
import json


def ids_to_multinomial(id, categories):
    id_to_idx = {id: index for index, id in enumerate(categories)}
    return id_to_idx[id]


def pickle_load(path):
    with open(path, 'rb') as fin:
        data = pickle.load(fin)

    return data


class AVQA_dataset(Dataset):

    def __init__(self, args, label, audios_feat_dir, visual_feat_dir,
                 clip_vit_b32_dir, clip_qst_dir, clip_word_dir,
                 transform=None, mode_flag='train'):

        self.args = args

        samples = json.load(open('./dataset/split_que_id/music_avqa_train.json', 'r'))

        # Question
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])
        # ques_vocab.append('fifth')

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 14  # question length

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir

        self.clip_vit_b32_dir = clip_vit_b32_dir
        self.clip_qst_dir = clip_qst_dir
        self.clip_word_dir = clip_word_dir

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']

        try:
            audios_feat = np.load(os.path.join(self.audios_feat_dir, name + '.npy'))
        except:
            audios_feat = np.zeros(shape=(60, 128))

        visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '_image_feats.npy'))
        needs_pad = visual_CLIP_feat.shape[0]
        if 60 - needs_pad > 0:
            visual_feat = np.pad(visual_CLIP_feat[:60, :], ((60 - needs_pad, 0), (0, 0)))
        else:
            visual_feat = visual_CLIP_feat[:60, :]

        visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '_patch_feats.npy'))
        needs_pad = visual_CLIP_feat.shape[0]
        if 60 - needs_pad > 0:
            patch_feat = np.pad(visual_CLIP_feat[:60, :, :], ((60 - needs_pad, 0), (0, 0), (0, 0)))
        else:
            patch_feat = visual_CLIP_feat[:60, :, :]

        word_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_words.npy'))
        question_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_sent.npy'))

        ### answer
        answer = sample['anser']
        answer_label = ids_to_multinomial(answer, self.ans_vocab)
        answer_label = torch.from_numpy(np.array(answer_label)).long()

        sample = {'video_name': name,
                  'audios_feat': audios_feat,
                  'visual_feat': visual_feat,
                  'patch_feat': patch_feat,
                  'question': question_feat,
                  'qst_word': word_feat,
                  'answer_label': answer_label,
                  'question_id': question_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PerceptionAVQA_dataset(Dataset):
    def __init__(self, args, label, audios_feat_dir, visual_feat_dir, video_dir, ocr_dir,
                 clip_vit_b32_dir, clip_qst_dir, clip_word_dir, clip_word_ans_dir,
                 transform=None, mode_flag='train'):

        self.args = args
        samples = json.load(open(f'./dataset/split_que_id/perception_{mode_flag}.json', 'r'))

        # Question
        ques_vocab = ['<pad>']
        ans_vocab = []
        i = 0
        for sample in samples:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1
            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)
            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        self.samples = json.load(open(label, 'r'))
        self.max_len = 195  # question length

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir
        self.video_feat_dir = video_dir
        self.ocr_feat_dir = ocr_dir

        self.clip_vit_b32_dir = clip_vit_b32_dir
        self.clip_qst_dir = clip_qst_dir
        self.clip_word_dir = clip_word_dir
        self.clip_word_ans_dir = clip_word_ans_dir

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        name = sample['video_id']
        question_id = sample['question_id']
        answer = sample['answer_id']

        audios_feat = pickle_load(os.path.join(self.audios_feat_dir, name + '_audioseg_1.pkl'))
        ocr_feat = pickle_load(os.path.join(self.ocr_feat_dir, name + '_ocrseg_1.pkl'))['ocr'][0]
        video_feat = pickle_load(os.path.join(self.video_feat_dir, name + '_seg_1.pkl'))

        visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '_image_feats.npy'))
        needs_pad = visual_CLIP_feat.shape[0]
        if 60 - needs_pad > 0:
            visual_feat = np.pad(visual_CLIP_feat[:60, :], ((60 - needs_pad, 0), (0, 0)))
        else:
            visual_feat = visual_CLIP_feat[:60, :]

        visual_CLIP_feat = np.load(os.path.join(self.clip_vit_b32_dir, name + '_patch_feats.npy'))
        needs_pad = visual_CLIP_feat.shape[0]
        if 60 - needs_pad > 0:
            patch_feat = np.pad(visual_CLIP_feat[:60, :, :], ((60 - needs_pad, 0), (0, 0), (0, 0)))
        else:
            patch_feat = visual_CLIP_feat[:60, :, :]

        word_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_words.npy'))
        question_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_sent.npy'))
        options_feat = np.load(os.path.join(self.clip_word_ans_dir, str(question_id) + '_sent.npy'))

        sample = {'video_name': name,
                  'audios_feat': audios_feat,
                  'ocr_feat': ocr_feat,
                  'video_feat': video_feat,
                  'visual_feat': visual_feat,
                  'patch_feat': patch_feat,
                  'question': question_feat,
                  'qst_word': word_feat,
                  'options_feat': options_feat,
                  'answer_feat': answer,
                  'question_id': question_id}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):
        video_name = sample['video_name']
        audios_feat = sample['audios_feat']
        ocr_feat = sample['ocr_feat']
        video_feat = sample['video_feat']
        visual_feat = sample['visual_feat']
        patch_feat = sample['patch_feat']
        question = sample['question']
        qst_word = sample['qst_word']
        options_feat = sample['options_feat']
        answer_feat = sample['answer_feat']
        question_id = sample['question_id']


        batch = {
            'audios_feat': audios_feat.float(),
            'ocr_feat': ocr_feat.float(),
            'video_feat': video_feat.float(),
            'visual_feat': torch.from_numpy(visual_feat).float(),
            'patch_feat': torch.from_numpy(patch_feat).float(),
            'question': torch.from_numpy(question[0]).float(),
            'qst_word': torch.from_numpy(qst_word[0]).float(),
            'options_feat': torch.from_numpy(options_feat).float(),
            'answer_label': torch.LongTensor([answer_feat])
        }



        return batch
        # 'video_name': video_name,
        # 'question_id': question_id
