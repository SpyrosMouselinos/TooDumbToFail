import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import *
from nets.PSTP_Net_Cat import Cataphract
from configs.arguments_PSTP_Cat import parser
import warnings
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')




def train(args, model, train_loader, optimizer, writer, epoch, scaler=None):
    model.train()

    running_train_loss = 0
    running_train_acc = 0
    for batch_idx, sample in enumerate(train_loader):
        audios_feat = sample['audios_feat'].to('cuda')
        ocr_feat = sample['ocr_feat'].to('cuda')
        video_feat = sample['video_feat'].to('cuda')
        visual_feat = sample['visual_feat'].to('cuda')
        patch_feat = sample['patch_feat'].to('cuda')
        question = sample['question'].to('cuda').unsqueeze(1)
        qst_word = sample['qst_word'].to('cuda').unsqueeze(1)
        options_feat = sample['options_feat'].to('cuda')
        answer_label = sample['answer_label'].to('cuda')


        optimizer.zero_grad()
        if scaler is not None:
            with autocast(dtype=torch.float16):
                output_qa, loss, metric = model(audio=audios_feat, visual=visual_feat, patch=patch_feat,
                                                video=video_feat,
                                                ocr=ocr_feat, question=question, qst_word=qst_word,
                                                options=options_feat, answer=answer_label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_qa, loss, metric = model(audio=audios_feat, visual=visual_feat, patch=patch_feat, video=video_feat,
                                            ocr=ocr_feat, question=question, qst_word=qst_word,
                                            options=options_feat, answer=answer_label)
            loss.backward()
            optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += metric.item()

        if batch_idx % args.log_interval == 0:
            writer.add_scalar('run/train_loss', running_train_loss / (batch_idx + 1),
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('run/train_acc', running_train_acc / (batch_idx + 1),
                              epoch * len(train_loader) + batch_idx)
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                epoch, 100. * batch_idx / len(train_loader), running_train_loss / (batch_idx + 1),
                       running_train_acc / (batch_idx + 1)))


def eval(model, val_loader, writer, epoch, scaler=None):
    model.eval()
    total_loss = 0
    total_metric = 0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audios_feat = sample['audios_feat'].to('cuda')
            ocr_feat = sample['ocr_feat'].to('cuda')
            video_feat = sample['video_feat'].to('cuda')
            visual_feat = sample['visual_feat'].to('cuda')
            patch_feat = sample['patch_feat'].to('cuda')
            question = sample['question'].to('cuda')
            qst_word = sample['qst_word'].to('cuda')
            options_feat = sample['options_feat'].to('cuda')
            answer_label = sample['answer_label'].to('cuda')

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    _, loss, metric = model(audio=audios_feat, visual=visual_feat, patch=patch_feat,
                                            video=video_feat,
                                            ocr=ocr_feat, question=question, qst_word=qst_word,
                                            options=options_feat, answer=answer_label)
            else:
                _, loss, metric = model(audio=audios_feat, visual=visual_feat, patch=patch_feat,
                                        video=video_feat,
                                        ocr=ocr_feat, question=question, qst_word=qst_word,
                                        options=options_feat, answer=answer_label)

            total_examples += 1
            total_loss += loss.item()
            total_metric += metric.item()

        val_loss = (total_loss / total_examples)
        val_acc = (total_metric / total_examples)
        print('Current Loss: %.3f %%' % val_loss)
        print('Current Acc: %.3f %%' % val_acc)
    writer.add_scalar('metric/val_loss_qa', val_loss, epoch)
    writer.add_scalar('metric/val_acc_qa', val_acc, epoch)

    return val_acc


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    tensorboard_name = args.checkpoint
    writer = SummaryWriter('runs/strn/' + TIMESTAMP + '_' + tensorboard_name)

    from thop import profile
    from thop import clever_format

    model = Cataphract(args)
    model = model.to('cuda')
    audio_input = torch.randn(1, 768).to('cuda')
    visual_input = torch.randn(1, 60, 512).to('cuda')
    patch_input = torch.randn(1, 60, 50, 768).to('cuda')
    video_input = torch.randn(1, 768).to('cuda')
    ocr_input = torch.randn(1, 768).to('cuda')
    question_input = torch.randn(1, 1, 512).to('cuda')  # Squeezed inside first block
    qst_word_input = torch.randn(1, 1, 77, 512).to('cuda')  # Squeezed inside third block
    options_input = torch.randn(1, 3, 512).to('cuda')
    answer = torch.zeros(1, dtype=torch.long).to('cuda')

    flops, params = profile(model, inputs=(
        audio_input, visual_input, patch_input, video_input, ocr_input, question_input, qst_word_input, options_input,
        answer))
    print("profile: ", flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print("clever: ", flops, params)

    # -------------> Computation costs end

    train_dataset = PerceptionAVQA_dataset(label=args.label_train,
                                           args=args,
                                           audios_feat_dir=args.audios_feat_dir,  # Path to Audio feats
                                           visual_feat_dir=None,
                                           video_dir=args.video_feat_dir,
                                           ocr_dir=args.ocr_feat_dir,
                                           clip_vit_b32_dir=args.clip_vit_b32_dir,  # Path to CLIP Video feats
                                           clip_qst_dir=None,
                                           clip_word_dir=args.clip_word_dir,  # Path to CLIP Question feats
                                           clip_word_ans_dir=args.clip_word_ans_dir,  # Path to CLIP Answer feats
                                           transform=transforms.Compose([ToTensor()]),
                                           mode_flag='train')
    val_dataset = PerceptionAVQA_dataset(label=args.label_val,
                                         args=args,
                                         audios_feat_dir=args.audios_feat_dir,  # Path to Audio feats
                                         visual_feat_dir=None,
                                         video_dir=args.video_feat_dir,
                                         ocr_dir=args.ocr_feat_dir,
                                         clip_vit_b32_dir=args.clip_vit_b32_dir,  # Path to CLIP Video feats
                                         clip_qst_dir=None,
                                         clip_word_dir=args.clip_word_dir,  # Path to CLIP Question feats
                                         clip_word_ans_dir=args.clip_word_ans_dir,  # Path to CLIP Answer feats
                                         transform=transforms.Compose([ToTensor()]),
                                         mode_flag='valid')
    # TODO: CHANGE THIS
    print("You train on val and val on train!")
    train_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False)

    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_acc = 0
    best_epoch = 0
    if args.use_mixed:
        scaler = GradScaler()
    else:
        scaler = None
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        train(args, model, train_loader, optimizer, writer, epoch=epoch, scaler=scaler)

        # Evaluate on validation set
        if epoch % args.eval_every == 0:
            scheduler.step(epoch)
            current_acc = eval(model, val_loader, writer, epoch, scaler=scaler)
            if current_acc >= best_acc:
                best_acc = current_acc
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")

        print("Best Acc: %.2f %%" % best_acc)
        print("Best Epoch: ", best_epoch)
        print("*" * 20)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
