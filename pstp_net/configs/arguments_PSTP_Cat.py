import os
import argparse

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of OCR-Audio-Visual Question Answering')


### ======================== Dataset Configs ==========================

base_path = "/home/spyros/Desktop/TooDumbToFail/data/"
server_path = "/home/spyros/Desktop/TooDumbToFail/data/PERCEPTION"

parser.add_argument("--audios_feat_dir", type=str, default=os.path.join(server_path, 'audio_frames'),
                    help="audio feat dir")
parser.add_argument("--ocr_feat_dir", type=str, default=os.path.join(server_path, 'ocr_frames'),
                    help="audio feat dir")
parser.add_argument("--video_feat_dir", type=str, default=os.path.join(server_path, 'video_frames'),
                    help="audio feat dir")
parser.add_argument("--clip_vit_b32_dir", type=str, default=os.path.join(server_path, 'clip_vit_b32'),
                    help="clip_vit-b32_dir")
parser.add_argument("--clip_word_dir", type=str, default=os.path.join(server_path, 'clip_word'),
                    help="clip_word")
parser.add_argument("--clip_word_ans_dir", type=str, default=os.path.join(server_path, 'clip_word_ans'),
                    help="clip_word")
parser.add_argument("--frames_dir", type=str, default=os.path.join(server_path, 'avqa-frames-1fps'),
                    help="video frames dir")
parser.add_argument("--ocr_rel_videos", type=str, default=os.path.join(base_path, 'ocr_rel_videos.json'),
                    help="ocr related questions")
parser.add_argument("--answer_keys", type=str, default=os.path.join(base_path, 'answer_keysword_overlap.json'),
                    help="answer keys")

### ======================== Label Configs ==========================
parser.add_argument("--label_train", type=str, default="./dataset/split_que_id/perception_train.json",
                    help="train csv file")
parser.add_argument("--label_val", type=str, default="./dataset/split_que_id/perception_valid.json",
                    help="val csv file")
parser.add_argument("--label_test", type=str, default="./dataset/split_que_id/perception_test.json",
                    help="test csv file")


### ======================== Model Configs ==========================

parser.add_argument("--question_encoder", type=str, default='CLIP', metavar='qe',
                    help="quesiton encoder, CLIP")
parser.add_argument("--visual_encoder", type=str, default='CLIP', metavar='ve',
                    help="visual encoder, CLIP")
parser.add_argument("--spatial_qst_encoder", type=str, default='CLIP', metavar='trqe',
                    help="quesiton encoder, CLIP or LSTM")
parser.add_argument("--spatial_vis_encoder", type=bool, default=True, metavar='sve',
                    help="Spatial regions selector module, Use CLIP")
parser.add_argument("--use_word", type=bool, default=True, metavar='uc',
                    help="word encoder module")
parser.add_argument("--temp_select", type=bool, default=True, metavar='tsm',
                    help="temporal segments selection module")
parser.add_argument("--segs", type=int, default=60, metavar='SEG',
                    help="temporal segment numbers segments")
parser.add_argument("--top_k", type=int, default=10, metavar='TK',
                    help="top K temporal segments")
parser.add_argument("--spat_select", type=bool, default=True, metavar='ssm',
                    help="spatio regions selection module")
parser.add_argument("--top_m", type=int, default=25, metavar='TM',
                    help="top M spatial regions")
parser.add_argument("--global_local", type=bool, default=True, metavar='glm',
                    help="global local perception module")
parser.add_argument("--temp_grd", type=bool, default=False, metavar='tgm',
                    help="temporal grounding module")
parser.add_argument("--num_layers", type=int, default=4, metavar='num_layers',
                    help="num_layers")


### ======================== Learning Configs ==========================
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--eval_every', type=int, default=5, metavar='eval_every',
                    help='eval every epochs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


### ======================== Save Configs ==========================
parser.add_argument( "--checkpoint", type=str, default='Test_No_Vid',
                    help="save model name")
parser.add_argument("--model_save_dir", type=str, default='/home/spyros/Desktop/TooDumbToFail/pstp_net/runs/models_pstp_cat/',
                    help="model save dir")
parser.add_argument("--mode", type=str, default='train',
                    help="with mode to use")


### ======================== Runtime Configs ==========================
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_workers', type=int, default=2,
                    help='num_workers number')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu device number')
parser.add_argument("--use_mixed", type=bool, default=True, metavar='use_mixed'
                    )