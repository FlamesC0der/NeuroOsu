import cv2

from settings import *
from data.capture_screen import capture_screen
from util.loader import save_data


def main():
    while True:
        frame = capture_screen()

        x, y = pyautogui.position()

        if x / w > 1:
            continue

        cv2.imshow('Computer vision', frame)
        cv2.waitKey(1)

        save_data(frame, x, y, w, h)


if __name__ == "__main__":
    main()
