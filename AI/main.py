import torch
import torch.nn.functional as F
import pyautogui
import numpy as np
import cv2
import keyboard

from settings import *
from data.capture_screen import capture_screen
from nn.model import OsuModel

model = OsuModel()
model.load_state_dict(torch.load('osu_model.pth'))
model.eval()

active = False


def toggle():
    global active
    active = not active
    print("Resumed" if active else "Paused")


keyboard.add_hotkey("ctrl+shift+a", toggle)


def main():
    while True:
        img = capture_screen()

        cv2.imshow('Computer vision', img)
        cv2.waitKey(1)

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img)

        if active:
            pred = model(img)

            x, y = F.softmax(pred).detach().numpy()[0]
            print(x, y)
            x, y = x * w, y * h
            print(x, y)

            pyautogui.moveTo(x, y, 0.1)


if __name__ == "__main__":
    main()
