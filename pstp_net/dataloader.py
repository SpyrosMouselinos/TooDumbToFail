import pickle
import numpy as np
import skimage
import torch
import os
from torch.utils.data import Dataset
import ast
import json
import decord as de

ctx = de.cpu(0)


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
        samples = json.load(
            open(f'/home/spyros/Desktop/TooDumbToFail/pstp_net/dataset/split_que_id/perception_{mode_flag}.json', 'r'))

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

        if self.audios_feat_dir is not None:
            audios_feat = pickle_load(os.path.join(self.audios_feat_dir, name + '_audioseg_1.pkl'))
        else:
            audios_feat = None
        if self.ocr_feat_dir is not None:
            ocr_feat = pickle_load(os.path.join(self.ocr_feat_dir, name + '_ocrseg_1.pkl'))['ocr'][0]
        else:
            ocr_feat = None
        if self.video_feat_dir is not None:
            video_feat = pickle_load(os.path.join(self.video_feat_dir, name + '_seg_1.pkl'))
        else:
            video_feat = None

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
        if self.clip_word_dir is not None:
            word_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_words.npy'))
        else:
            word_feat = None
        if self.clip_qst_dir is not None:
            question_feat = np.load(os.path.join(self.clip_word_dir, str(question_id) + '_sent.npy'))
        else:
            question_feat = None

        if self.clip_word_ans_dir is not None:
            options_feat = np.load(os.path.join(self.clip_word_ans_dir, str(question_id) + '_sent.npy'))
        else:
            options_feat = None

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


class PerceptionBLIP_dataset(Dataset):
    def __init__(self,
                 args,
                 video_dir=None,
                 image_dir=None,
                 image_feat_dir=None,
                 question_feat_dir=None,
                 answer_feat_dir=None,
                 transform=None,
                 mode_flag='train'):

        self.args = args
        self.max_len = 195  # question length
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.image_feat_dir = image_feat_dir
        self.question_feat_dir = question_feat_dir
        self.answer_feat_dir = answer_feat_dir
        self.mode_flag = 'train'

        if self.image_feat_dir is not None and self.image_dir is None and self.video_dir is None:
            print("Reading pre-calculated BLIP2-Visual Features")
            self.visual_input_mode = 'feats'
        elif self.image_dir is not None:
            print("Reading directly from sets of images")
            from transformers import Blip2Processor, Blip2VisionModel
            self.visual_input_mode = 'images'
            self.i_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
            self.i_feat_extractor = Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to('cuda').half()
        elif self.video_dir is not None:
            print("Reading directly from videos")
            from transformers import Blip2Processor, Blip2VisionModel
            self.visual_input_mode = 'videos'
            self.i_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
            self.i_feat_extractor = Blip2VisionModel.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to('cuda').half()
        else:
            raise ValueError()

        if self.question_feat_dir is not None:
            print("Reading pre-calculated BLIP2-Text Question Features")
            self.text_qinput_mode = 'feats'
        else:
            print("Reading directly questions from json")
            self.text_qinput_mode = 'text'
            from transformers import Blip2Processor
            self.text_qprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

        if self.answer_feat_dir is not None:
            print('Reading pre-calculated BLIP2-Text Answer Features')
            self.text_ainput_mode = 'feats'
        else:
            print("Reading directly answers from json")
            self.text_ainput_mode = 'text'
            from transformers import Blip2Processor
            if self.text_qinput_mode == 'text':
                self.text_aprocessor = self.text_qprocessor
            else:
                self.text_aprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")

        self.set_to_train_mode()
        self.transform = transform

    def set_to_train_mode(self):
        self.mode_flag = 'train'
        self.samples = json.load(
            open(f'/home/spyros/Desktop/TooDumbToFail/pstp_net/dataset/split_que_id/perception_{self.mode_flag}.json',
                 'r'))

    def set_to_val_mode(self):
        self.mode_flag = 'valid'
        self.samples = json.load(
            open(f'/home/spyros/Desktop/TooDumbToFail/pstp_net/dataset/split_que_id/perception_{self.mode_flag}.json',
                 'r'))

    def set_to_test_mode(self):
        self.mode_flag = 'test'
        self.samples = json.load(
            open(f'/home/spyros/Desktop/TooDumbToFail/pstp_net/dataset/split_que_id/perception_{self.mode_flag}.json',
                 'r'))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def pad_collate(batch):
        final_batch = {}
        final_batch.update({"image_feats": torch.stack([f['image_feats'][:3, :, :] for f in batch], dim=0)})
        qt = torch.nn.utils.rnn.pad_sequence([f['question_and_options_tokenized'] for f in batch], batch_first=True,
                                             padding_value=1)
        final_batch.update({"question_and_options_tokenized": qt})
        if 'answer_tokenized' in batch[0]:
            at = torch.nn.utils.rnn.pad_sequence([f['answer_tokenized'] for f in batch], batch_first=True, padding_value=1)
            final_batch.update({"answer_tokenized": at})
        return final_batch

    def read_video(self, path):
        if not os.path.isfile(path):
            raise ValueError('Given path does not lead to a file but a folder!')

        video_basename, video_extension = os.path.splitext(path)
        image_folder = video_basename
        video_id = video_basename.split('/')[-1]
        os.makedirs(image_folder, exist_ok=True)
        avr = de.VideoReader(uri=path, ctx=ctx, width=384, height=384,
                             num_threads=0)
        t = len(avr)
        indices = np.linspace(0, t - 2, 60)
        indices = np.clip(indices, 0, t - 2).astype(int)
        frames = avr.get_batch(indices).asnumpy()
        for frame_index in indices:
            frame_filename = f"{video_id}_frame{frame_index}.jpg"
            frame_path = os.path.join(image_folder, frame_filename)
            skimage.io.imsave(frame_path, frames[frame_index])
        return frames

    def read_set_of_images(self, path):
        video_basename, video_extension = os.path.splitext(path)
        image_folder = video_basename
        frames = []
        for frame_filename in os.listdir(image_folder):
            frame_path = os.path.join(image_folder, frame_filename)
            frames.append(skimage.io.imread(frame_path))
        return frames

    def convert_set_of_images_to_feats(self, set_of_images, path, process_batch_size=8):
        tensor_images = torch.Tensor(set_of_images)
        total_images = tensor_images.size()[0]
        batches = total_images // process_batch_size
        leftover = total_images - (process_batch_size * batches)
        vision_outputs = []
        with torch.no_grad():
            for i in range(batches):
                out = self.i_feat_extractor(
                    pixel_values=tensor_images[i:(i + 1) * process_batch_size, :, :, :].to('cuda', torch.float16))[0]
                for j in out:
                    vision_outputs.append(j)
            if leftover > 0:
                out = self.i_feat_extractor(pixel_values=tensor_images[-leftover:, :, :, :].to('cuda', torch.float16))[
                    0]
                for j in out:
                    vision_outputs.append(j)
        vision_outputs = torch.stack(vision_outputs, dim=0).cpu()
        with open(path + '.pkl', 'wb') as fout:
            pickle.dump(vision_outputs, fout)
        ### Clean up ###
        del tensor_images
        del out
        ### End of Clean Up ###
        return vision_outputs

    def read_image_feats(self, path):
        if os.path.exists(path + '.pkl'):
            with open(path + '.pkl', 'rb') as fout:
                vision_outputs = pickle.load(fout)
            return vision_outputs
        else:
            raise ValueError(f'Image features for: {path} not found!')

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        question_text = sample['question_content']
        options_text = sample['options']
        if self.mode_flag != 'test':
            answer_index = sample['answer_id']

        if self.text_qinput_mode == 'text':
            question_and_options_tokenized = 'Question: ' + question_text + '\n'
            question_and_options_tokenized += 'Option A: ' + options_text[0] + '\n'
            question_and_options_tokenized += 'Option B: ' + options_text[1] + '\n'
            question_and_options_tokenized += 'Option C: ' + options_text[2] + '\n Answer: '
            question_and_options_tokenized = self.text_qprocessor(text=question_and_options_tokenized)['input_ids']
            if self.mode_flag != 'test':
                answer_prefix = f'{["A", "B", "C"][answer_index]}'
                answer_tokenized = self.text_aprocessor(text=answer_prefix)['input_ids']
        else:
            raise NotImplementedError

        if self.visual_input_mode == 'videos':
            # Check if they exist already #
            if os.path.exists(os.path.join(self.image_feat_dir, video_id) + '.pkl'):
                with open(os.path.join(self.image_feat_dir, video_id) + '.pkl', 'rb') as fout:
                    image_feats = pickle.load(fout)
            else:
                set_of_images = self.read_video(path=os.path.join(self.video_dir, video_id, video_id + '.mp4'))
                images_tokenized = self.i_processor(images=set_of_images)['pixel_values']
                image_feats = self.convert_set_of_images_to_feats(images_tokenized,
                                                                  os.path.join(self.image_feat_dir, video_id),
                                                                  process_batch_size=1)
        elif self.visual_input_mode == 'images':
            if os.path.exists(os.path.join(self.image_feat_dir, video_id) + '.pkl'):
                with open(os.path.join(self.image_feat_dir, video_id) + '.pkl', 'rb') as fout:
                    image_feats = pickle.load(fout)
            else:
                set_of_images = self.read_set_of_images(path=os.path.join(self.video_dir, video_id, video_id))
                images_tokenized = self.i_processor(images=set_of_images)['pixel_values']
                image_feats = self.convert_set_of_images_to_feats(images_tokenized,
                                                                  os.path.join(self.image_feat_dir, video_id),
                                                                  process_batch_size=self.args.preprocess_batch_size)
        elif self.visual_input_mode == 'feats':
            image_feats = self.read_image_feats(os.path.join(self.image_feat_dir, video_id))
        else:
            raise NotImplementedError

        sample = {'image_feats': image_feats,
                  'question_and_options_tokenized': question_and_options_tokenized
                  }

        if self.mode_flag != 'test':
            sample.update({'answer_tokenized': answer_tokenized})
            sample.update({'answer_index': answer_index})
        if self.transform:
            sample = self.transform(sample)

        return sample


class TensorHalfMove(object):
    def __call__(self, sample):
        image_feats = sample['image_feats']
        question_and_options_tokenized = sample['question_and_options_tokenized']
        if 'answer_tokenized' in sample:
            answer_tokenized = sample['answer_tokenized']
            answer_index = sample['answer_index']

        batch = {
            'image_feats': image_feats,
            'question_and_options_tokenized': torch.LongTensor(question_and_options_tokenized),
        }
        if 'answer_tokenized' in sample:
            batch.update({'answer_tokenized': torch.LongTensor(answer_tokenized)})
            batch.update({'answer_index': torch.LongTensor([answer_index])})

        return batch


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
            'visual_feat': torch.from_numpy(visual_feat).float(),
            'patch_feat': torch.from_numpy(patch_feat).float(),
            'answer_label': torch.LongTensor([answer_feat])
        }
        if audios_feat is not None:
            batch.update({'audios_feat': audios_feat.float()})
        if ocr_feat is not None:
            batch.update({'ocr_feat': ocr_feat.float()})
        if video_feat is not None:
            batch.update({'video_feat': video_feat.float()})
        if options_feat is not None:
            batch.update({'options_feat': torch.from_numpy(options_feat).float()})
        if question is not None:
            batch.update({'question': torch.from_numpy(question[0]).float()})
        if qst_word is not None:
            batch.update({'qst_word': torch.from_numpy(qst_word[0]).float()})
        return batch
