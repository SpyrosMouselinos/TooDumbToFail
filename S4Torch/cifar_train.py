import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from s4torch.model import S4Model
from torch.optim import AdamW
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cs(batch):
    images = []
    labels = []
    for item in batch:
        C, H, W = item[0].size()
        images.append(item[0].view(C, H * W).transpose(1, 0).contiguous())
        labels.append(item[1])
    return {'image': torch.stack(images, dim=0), 'label': torch.LongTensor(labels)}


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))

train_dataloader = DataLoader(trainset, shuffle=True, batch_size=32, collate_fn=cs)
eval_dataloader = DataLoader(testset, batch_size=32, collate_fn=cs)

model = S4Model(d_input=3, d_model=128, d_output=10, n_blocks=6, n=64, l_max=32 * 32, wavelet_tform=False,
                collapse=True, p_dropout=0.2, pooling=None)

optimizer = AdamW(model.parameters(), lr=1e-3)
cel = CrossEntropyLoss()
metric = evaluate.load("accuracy")
model.to(device)

progress_bar = tqdm(range(len(train_dataloader)))

model.train()
for epoch in range(100):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['image'])
        loss = cel(outputs, batch['label'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        results = metric.compute(references=batch['label'], predictions=outputs.argmax(dim=1))
        print(f"Training Acc: {results['accuracy']}")


    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["label"])

    metric.compute()
