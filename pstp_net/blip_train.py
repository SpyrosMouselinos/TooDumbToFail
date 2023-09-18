import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import *
from nets.Temporal_Net import TBLIP
from configs.arguments_BLIP_Cat import parser
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
        image_feats = sample['image_feats'].to('cuda')
        question_and_options_tokenized = sample['question_and_options_tokenized'].to('cuda')
        answer_tokenized = sample['answer_tokenized'].to('cuda')
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(dtype=torch.float16):
                output_qa, loss, metric = model(image_feats=image_feats,
                                                question_and_options_tokenized=question_and_options_tokenized,
                                                answer_tokenized=answer_tokenized)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output_qa, loss, metric = model(image_feats=image_feats,
                                            question_and_options_tokenized=question_and_options_tokenized,
                                            answer_tokenized=answer_tokenized)
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
            image_feats = sample['image_feats'].to('cuda')
            question_and_options_tokenized = sample['question_and_options_tokenized'].to('cuda')
            answer_tokenized = sample['answer_tokenized'].to('cuda')

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
    writer.add_scalar('metric/val_loss_qa', val_loss, epoch)
    writer.add_scalar('metric/val_acc_qa', val_acc, epoch)

    return val_acc

def m_test(model, test_loader, scaler=None):
    suggested_answers = []
    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            image_feats = sample['image_feats'].to('cuda')
            question_and_options_tokenized = sample['question_and_options_tokenized'].to('cuda')

            if scaler is not None:
                with autocast(dtype=torch.float16):
                    suggested_answer, _, _ = model(image_feats=image_feats,
                                            question_and_options_tokenized=question_and_options_tokenized,
                                            answer_tokenized=None)
            else:
                suggested_answer, _, _ = model(image_feats=image_feats,
                                        question_and_options_tokenized=question_and_options_tokenized,
                                        answer_tokenized=None)
    suggested_answers.append(suggested_answer)
    return suggested_answers

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)

    tensorboard_name = args.checkpoint
    writer = SummaryWriter('runs/strn/' + TIMESTAMP + '_' + tensorboard_name)

    model = TBLIP.from_pretrained("Salesforce/blip2-opt-2.7b")
    #model = model.to('cuda').half()

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
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        collate_fn=dataset.pad_collate)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0
    best_epoch = 0
    if args.use_mixed:
        scaler = GradScaler()
    else:
        scaler = None
    for epoch in range(1, args.epochs + 1):
        train(args, model, loader, optimizer, writer, epoch=epoch, scaler=scaler)
        scheduler.step(epoch)

        # Evaluate on validation set
        if epoch % args.eval_every == 0:
            dataset.set_to_val_mode()
            current_acc = eval(model, loader, writer, epoch, scaler=scaler)
            if current_acc >= best_acc:
                best_acc = current_acc
                best_epoch = epoch
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
            dataset.set_to_train_mode()

        print("Best Acc: %.2f %%" % best_acc)
        print("Best Epoch: ", best_epoch)
        print("*" * 20)


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    main()
