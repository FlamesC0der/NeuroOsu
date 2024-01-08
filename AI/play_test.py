import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pyautogui

from settings import w, h
from nn.model import OsuModel

model = OsuModel()
model.load_state_dict(torch.load('osu_model.pth'))
model.eval()

img = cv2.imread('/Users/alexey/PycharmProjects/NeuroOsu/test_data/frames/68.png', cv2.IMREAD_GRAYSCALE)

img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=0)

img = img.astype(np.float32) / 255.0
img = torch.from_numpy(img)

pred = model(img)

x, y = F.softmax(pred).detach().numpy()[0]
print(x, y)
x, y = x * w, y * h
print(x, y)

pyautogui.click(x, y)
