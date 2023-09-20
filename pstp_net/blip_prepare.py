import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataloader import device, PerceptionBLIP_dataset, TensorHalfMove
from configs.arguments_BLIP_Cat import parser
from torch.cuda.amp import autocast


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
    for mode in ['train', 'valid', 'test']:
        dataset = PerceptionBLIP_dataset(args=args,
                                         video_dir=args.video_dir,
                                         image_dir=args.image_dir,
                                         image_feat_dir=args.image_feat_dir,
                                         question_feat_dir=None,
                                         answer_feat_dir=None,
                                         transform=transforms.Compose([TensorHalfMove()]),
                                         mode_flag=mode)
        if mode == 'train':
            dataset.set_to_train_mode()
        elif mode == 'valid':
            dataset.set_to_val_mode()
        elif mode == 'test':
            dataset.set_to_test_mode()

        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=dataset.pad_collate)
        for idx, _ in enumerate(loader):
            print(f"Processed {idx} / {len(loader)} {mode} items.")


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    t_main()
