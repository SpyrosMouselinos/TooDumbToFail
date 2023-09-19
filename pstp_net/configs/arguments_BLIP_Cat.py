import os
import argparse
import torch

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of OCR-Audio-Visual Question Answering')
device = 'cuda'
dtype = torch.float16 if device == 'cuda' else torch.float32
### ======================== Dataset Configs ==========================
c_path = '/'.join(os.getcwd().split('/')[:-1])
base_path = c_path + "/data/"
server_path = base_path + "/PERCEPTION"
parser.add_argument("--video_dir", type=str, default=os.path.join(server_path, 'avqa-frames-1fps/'))
parser.add_argument("--image_dir", type=str, default=None)
parser.add_argument("--image_feat_dir", type=str, default=os.path.join(server_path, 'blip_feats/'))
parser.add_argument("--question_feat_dir", type=str, default=None)
parser.add_argument("--answer_feat_dir", type=str, default=None)
parser.add_argument("--preprocess_batch_size", type=int, default=4)

parser.add_argument("--answer_keys", type=str, default=os.path.join(base_path, 'answer_keysword_overlap.json'),
                    help="answer keys")

### ======================== Label Configs ==========================
parser.add_argument("--label_train", type=str,
                    default=c_path + "/pstp_net/dataset/split_que_id/perception_train.json",
                    help="train csv file")
parser.add_argument("--label_val", type=str,
                    default=c_path + "/pstp_net/dataset/split_que_id/perception_valid.json",
                    help="val csv file")
parser.add_argument("--label_test", type=str,
                    default=c_path + "/pstp_net/dataset/split_que_id/perception_test.json",
                    help="test csv file")

### ======================== Learning Configs ==========================
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--eval_every', type=int, default=2, metavar='eval_every',
                    help='eval every epochs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

### ======================== Save Configs ==========================
parser.add_argument("--checkpoint", type=str, default='BLIP2',
                    help="save model name")
parser.add_argument("--model_save_dir", type=str, default=c_path + '/pstp_net/runs/',
                    help="model save dir")
parser.add_argument("--mode", type=str, default='train',
                    help="with mode to use")

### ======================== Runtime Configs ==========================
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_workers', type=int, default=0,
                    help='num_workers number')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device number')
parser.add_argument("--use_mixed", type=bool, default=True, metavar='use_mixed'
                    )
