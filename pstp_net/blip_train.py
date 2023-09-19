import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataloader import device, PerceptionBLIP_dataset, TensorHalfMove
from nets.Temporal_Net import TBLIP
from configs.arguments_BLIP_Cat import parser
import warnings
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
warnings.filterwarnings('ignore')


def train(args, model, train_loader, optimizer, epoch, scaler=None):
    model.train()

    running_train_loss = 0
    running_train_acc = 0
    for batch_idx, sample in enumerate(train_loader):
        image_feats = sample['image_feats'].to(device).float()
        question_and_options_tokenized = sample['question_and_options_tokenized'].to(device)
        answer_tokenized = sample['answer_tokenized'].to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(dtype=torch.float16):
                output_qa, loss, metric = model(image_feats=image_feats,
                                                question_and_options_tokenized=question_and_options_tokenized,
                                                answer_tokenized=answer_tokenized)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
        else:
            output_qa, loss, metric = model(image_feats=image_feats,
                                            question_and_options_tokenized=question_and_options_tokenized,
                                            answer_tokenized=answer_tokenized)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        running_train_loss += loss.item()
        running_train_acc += metric.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tAccuracy: {:.3f}'.format(
                epoch, 100. * batch_idx / len(train_loader), running_train_loss / (batch_idx + 1),
                       running_train_acc / (batch_idx + 1)))


def eval(model, val_loader, epoch, scaler=None):
    model.eval()
    total_loss = 0
    total_metric = 0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            image_feats = sample['image_feats'].to(device)
            question_and_options_tokenized = sample['question_and_options_tokenized'].to(device)
            answer_tokenized = sample['answer_tokenized'].to(device)

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    _, loss, metric = model(image_feats=image_feats,
                                            question_and_options_tokenized=question_and_options_tokenized,
                                            answer_tokenized=answer_tokenized)
            else:
                _, loss, metric = model(image_feats=image_feats,
                                        question_and_options_tokenized=question_and_options_tokenized,
                                        answer_tokenized=answer_tokenized)

            total_examples += 1
            total_loss += loss.item()
            total_metric += metric.item()

        val_loss = (total_loss / total_examples)
        val_acc = (total_metric / total_examples)
        print('Current Loss: %.3f %%' % val_loss)
        print('Current Acc: %.3f %%' % val_acc)
    return val_acc




def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    if device != 'cpu':
        model = TBLIP.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    else:
        model = TBLIP.from_pretrained("Salesforce/blip2-opt-2.7b")
    dataset = PerceptionBLIP_dataset(args=args,
                                     video_dir=args.video_dir,
                                     image_dir=args.image_dir,
                                     image_feat_dir=args.image_feat_dir,
                                     question_feat_dir=None,
                                     answer_feat_dir=None,
                                     transform=transforms.Compose([TensorHalfMove()]),
                                     mode_flag='train')

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        collate_fn=dataset.pad_collate)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = -1
    best_epoch = 0
    if args.use_mixed:
        scaler = GradScaler()
    else:
        scaler = None
    for epoch in range(1, args.epochs + 1):
        train(args, model, loader, optimizer, epoch=epoch, scaler=scaler)
        scheduler.step(epoch)

        # Evaluate on validation set
        if epoch % args.eval_every == 0:
            dataset.set_to_val_mode()
            current_acc = eval(model, loader, epoch, scaler=scaler)
            if current_acc >= best_acc:
                best_acc = current_acc
                best_epoch = epoch
                model.save_model(
                    args.model_save_dir + args.checkpoint + f"_epoch_{epoch}_acc_{int(100 * best_acc)}" + ".pt")
            dataset.set_to_train_mode()

        print("Best Acc: %.2f %%" % best_acc)
        print("Best Epoch: ", best_epoch)
        print("*" * 20)




if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
