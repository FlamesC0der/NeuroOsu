import torch
import torch.nn as nn
import torch.nn.functional as F
import pyautogui
import numpy as np
import cv2

from settings import *
from data.capture_screen import capture_screen
from nn.model import OsuModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OsuModel().to(device)
    model.load_state_dict(torch.load('osu_model.pth'))
    model.eval()

    img = capture_screen()

    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)

    img = torch.from_numpy(img)

    pred = model(img)

    x, y = F.softmax(pred).detach().numpy()[0]
    print(x, y)
    x, y = x * w, y * h
    print(x, y)
    pyautogui.click(x, y)


if __name__ == "__main__":
    main()
