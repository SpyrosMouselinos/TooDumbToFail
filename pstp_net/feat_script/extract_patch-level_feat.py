import os
import torch
from PIL import Image
import numpy as np
import glob

from pstp_net.feat_script.clip import load as clip_load

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_load("ViT-B/32", device=device)


def clip_feat_extract(img):
    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def ImageClIP_feat_extract(dir_fps_path, dst_clip_path):
    video_list = os.listdir(dir_fps_path)
    video_idx = 0
    total_nums = len(video_list)

    for video in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video)

        save_file_img = os.path.join(dst_clip_path, video +  '_image_feats.npy')
        save_file_patch = os.path.join(dst_clip_path, video + '_patch_feats.npy')
        if os.path.exists(save_file_img) and os.path.exists(save_file_patch):
            print(video + '.npy', "is already processed!")
            continue

        video_img_list = sorted(glob.glob(os.path.join(dir_fps_path, video, video, '*.jpg')))

        params_frames = len(video_img_list)
        samples = np.round(np.linspace(0, params_frames - 1, params_frames))

        img_list = [video_img_list[int(sample)] for sample in samples]
        patch_features = torch.zeros(len(img_list), 50, 768)
        img_features = torch.zeros(len(img_list), 512)

        idx = 0
        for img_cont in img_list:
            img_idx_feat = clip_feat_extract(img_cont)
            img_features[idx] = img_idx_feat[0]
            patch_features[idx] = img_idx_feat[1]
            idx += 1

        img_features = img_features.float().cpu().numpy()
        patch_features = patch_features.float().cpu().numpy()
        np.save(save_file_img, img_features)
        np.save(save_file_patch, patch_features)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx)


if __name__ == "__main__":
    dir_fps_path = '../../data/MUSIC-AVQA/avqa-frames-1fps'
    dst_clip_path = '../../data/MUSIC-AVQA/clip_vit_b32'

    ImageClIP_feat_extract(dir_fps_path, dst_clip_path)
