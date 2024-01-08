import cv2
import time
import keyboard

from settings import *
from data.capture_screen import capture_screen
from util.loader import save_data

active = False


def toogle():
    global active
    active = not active


keyboard.add_hotkey("ctrl+shift+a", toogle)


def main():
    mode = input("select mode (t/v)")
    path = TRAIN_DIR if "t" in mode else TEST_DIR
    print(path)
    while True:
        if active:
            frame = capture_screen()

            x, y = pyautogui.position()

            if x / w > 1:
                continue

            cv2.imshow('Computer vision', frame)
            cv2.waitKey(1)

            save_data(path, frame, x, y, w, h)


if __name__ == "__main__":
    main()
