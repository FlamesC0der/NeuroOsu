import torch
import torch.nn.functional as F
import pyautogui
import numpy as np
import cv2

from settings import *
from data.capture_screen import capture_screen
from nn.model import OsuModel

model = OsuModel()
model.load_state_dict(torch.load('osu_model.pth'))
model.eval()

c = 0


def main():
    global c
    while True:
        img = capture_screen()

        cv2.imshow('Computer vision', img)
        cv2.waitKey(1)

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img)

        pred = model(img)

        x, y = F.softmax(pred).detach().numpy()[0]
        print(x, y)
        x, y = x * w, y * h
        print(x, y)

        pyautogui.click(x, y)

        c += 1

        if c >= 100:
            break


if __name__ == "__main__":
    main()
