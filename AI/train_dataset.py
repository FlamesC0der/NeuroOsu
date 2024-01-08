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
    OsuDataset('train'),
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


epochs = 20

print(len(train_loader))

train_acc_values = []
train_loss_values = []

print("\nstarted training\n")

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']

        optimizer.zero_grad()
        # label = F.one_hot(label, 2)

        pred = model(img)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        loss_val += loss_item

        acc_current = accuracy(pred, label)
        acc_val += acc_current

        pbar.set_description(f'Epoch [{epoch + 1}/{epochs}] loss: {loss_item:.5f}, accuracy: {acc_current:.3f}')

    avg_loss = loss_val / len(train_loader)
    avg_acc = acc_val / len(train_loader)

    train_loss_values.append(avg_loss)
    train_acc_values.append(avg_acc)

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot(range(1, epochs + 1), train_acc_values, label="Training Accuracy", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Loss', color=color)
ax2.plot(range(1, epochs + 1), train_loss_values, label="Training Loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Training Accuracy and Loss')
plt.show()


torch.save(model.state_dict(), 'osu_model.pth')
