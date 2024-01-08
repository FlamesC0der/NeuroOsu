import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from nn.model import OsuModel
from nn.dataset import OsuDataset

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    OsuDataset('train'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    OsuDataset('test'),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

model = OsuModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)

    return answer.mean()


epochs = 10

print(len(train_loader))

train_acc_values = []
train_loss_values = []
val_loss_values = []
val_acc_values = []

print("started training\n")

for epoch in range(epochs):
    # train

    tr_loss = 0
    tr_acc = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']

        optimizer.zero_grad()
        # label = F.one_hot(label, 2)

        pred = model(img)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        tr_loss += loss_item

        acc = accuracy(pred, label)
        tr_acc += acc

        pbar.set_description(f'Training Epoch [{epoch + 1}/{epochs}] loss: {loss_item:.5f}, accuracy: {acc:.3f}')

    avg_loss = tr_loss / len(train_loader)
    avg_acc = tr_acc / len(train_loader)

    # test

    val_loss = 0
    val_acc = 0
    model.eval()

    with torch.no_grad():
        for sample in (pbar := tqdm(test_loader)):
            img, label = sample['img'], sample['label']

            pred = model(img)
            loss = loss_fn(pred, label)

            loss_item = loss.item()
            val_loss += loss.item()

            acc = accuracy(pred, label)
            val_acc += acc

            pbar.set_description(f'Validating Epoch [{epoch + 1}/{epochs}] loss: {loss_item:.5f}, accuracy: {acc:.3f}')

    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader)

    train_loss_values.append(avg_loss)
    train_acc_values.append(avg_acc)
    val_loss_values.append(avg_val_loss)
    val_acc_values.append(avg_val_acc)


# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_acc_values, label="Training Accuracy", color='tab:red')
plt.plot(range(1, epochs + 1), val_acc_values, label="Validation Accuracy", color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_loss_values, label="Training Loss", color='tab:red')
plt.plot(range(1, epochs + 1), val_loss_values, label="Validation Loss", color='tab:blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

torch.save(model.state_dict(), 'osu_model.pth')
