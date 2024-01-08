import mss
import numpy as np
import cv2

from AI.settings import w, h, MONITOR_POS


def capture_screen() -> np.ndarray:
    with mss.mss() as sct:
        frame = sct.grab(MONITOR_POS)
        frame = np.array(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (80, 60))
        frame = frame.astype(np.float32)
        frame = frame / 255.0

        return frame
