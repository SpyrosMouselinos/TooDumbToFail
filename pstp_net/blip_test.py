import json
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataloader import device, PerceptionBLIP_dataset, TensorHalfMove
from nets.Temporal_Net import TBLIP
from configs.arguments_BLIP_Cat import parser
import warnings
from datetime import datetime
from torch.cuda.amp import autocast

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')


def m_test(model, test_loader, scaler=None):
    suggested_answers = []
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            image_feats = sample['image_feats'].to(device)
            question_and_options_tokenized = sample['question_and_options_tokenized'].to(device)

            if scaler:
                with autocast(dtype=torch.float16):
                    suggested_answer, _, _ = model.answer(image_feats=image_feats,
                                                          question_and_options_tokenized=question_and_options_tokenized,
                                                          )
            else:
                suggested_answer, _, _ = model.answer(image_feats=image_feats,
                                                      question_and_options_tokenized=question_and_options_tokenized,
                                                      )
    suggested_answers.append(suggested_answer[0])
    return suggested_answers


def t_main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if device != 'cpu':
        model = TBLIP.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    else:
        model = TBLIP.from_pretrained("Salesforce/blip2-opt-2.7b")
    ### Find and load best acc model ###
    try:
        saved_model_path = os.listdir(args.model_save_dir)
        best_acc = 0
        best_ckp = None
        for f in saved_model_path:
            try:
                acc = int(f.split('_acc_')[1].split('.pt')[0])
            except:
                acc = 0

            if acc > best_acc:
                best_acc = acc
                best_ckp = f
        if not best_ckp is None:
            model.load_model(path=best_ckp)
            print(f"Loading model ckp: {best_ckp}")
    except:
        print(f"Path {args.model_save_dir} not found!")

    dataset = PerceptionBLIP_dataset(args=args,
                                     video_dir=args.video_dir,
                                     image_dir=args.image_dir,
                                     image_feat_dir=args.image_feat_dir,
                                     question_feat_dir=None,
                                     answer_feat_dir=None,
                                     transform=transforms.Compose([TensorHalfMove()]),
                                     mode_flag='test')
    dataset.set_to_test_mode()

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=dataset.pad_collate)

    suggested_answers = m_test(model, loader, scaler=args.use_mixed)
    answers = {}
    for idx, answer in zip(range(len(dataset)), suggested_answers):
        sample = dataset.samples[idx]
        video_id = sample['video_id']
        options_text = sample['options']

        if video_id in answers:
            answers[video_id].append({
                'id': len(answers[video_id]),
                'answer_id': answer,
                'answer': options_text[answer]
            })
        else:
            answers.update({video_id: []})
            answers[video_id].append({
                'id': len(answers[video_id]),
                'answer_id': answer,
                'answer': options_text[answer]
            })

    with open('test_results_tblip.json', 'w') as my_file:
        json.dump(answers, my_file)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    t_main()
