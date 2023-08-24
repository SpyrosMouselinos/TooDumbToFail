import os
import torch
import numpy as np
import json
import ast

from pstp_net.feat_script.clip import load as clip_load
from pstp_net.feat_script.clip import tokenize as clip_tokenize
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip_load("ViT-B/32", device=device)


def qst_feat_extract(qst):

    text = clip_tokenize(qst).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)

    return text_features


def QstCLIP_feat(json_path, dst_qst_path):

    samples = json.load(open(json_path, 'r'))

    ques_vocab = ['<pad>']

    i = 0
    for sample in samples:
        i += 1
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        question_id = sample['question_id']
        #print("\n")
        #print("question id: ", question_id)

        sent_file = os.path.join(dst_qst_path, str(question_id) + '_sent.npy')
        words_file = os.path.join(dst_qst_path, str(question_id) + '_words.npy')

        if os.path.exists(sent_file) and os.path.exists(words_file):
            print(question_id, " already exists!")
            continue

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        for wd in question:
            if wd not in ques_vocab:
                ques_vocab.append(wd)

        question = ' '.join(question)
        #print(question)

        feats = qst_feat_extract(question)
        #print(qst_feat.shape)

        sent_features = feats[0].float().cpu().numpy()
        word_features = feats[1].float().cpu().numpy()

        np.save(sent_file, sent_features)
        np.save(words_file, word_features)



if __name__ == "__main__":

    json_path = "../dataset/split_que_id/music_avqa.json"
    dst_qst_path = "../../data/PERCEPTION/clip_word/"

    QstCLIP_feat(json_path, dst_qst_path)


