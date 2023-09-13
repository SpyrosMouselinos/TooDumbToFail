import os
import torch
import numpy as np
import json
import ast

from clip import load as clip_load
from clip import tokenize as clip_tokenize
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_load("ViT-B/32", device=device)


def ans_feat_extract(ans):
    text = clip_tokenize(ans).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


def AnsCLIP_feat(json_path, dst_ans_path):
    ids = []
    samples = json.load(open(json_path, 'r'))

    i = 0
    for sample in samples:
        i += 1
        options = [f.rstrip() for f in sample['options']]
        question_id = sample['question_id']
        ids.append(question_id)

        sent_file = os.path.join(dst_ans_path, str(question_id) + '_sent.npy')

        if os.path.exists(sent_file):
            print(question_id, " already exists!")
            continue


        feats = ans_feat_extract(options)
        sent_features = feats[0].float().cpu().numpy()
        np.save(sent_file, sent_features)




if __name__ == "__main__":

    json_path = "../dataset/split_que_id/perception.json"
    ans_qst_path = "../../data/PERCEPTION/clip_word_ans/"

    AnsCLIP_feat(json_path, ans_qst_path)


