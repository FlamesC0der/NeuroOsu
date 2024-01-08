import pyautogui

w, h = pyautogui.size()
MONITOR_POS = (300, 10, w - 300, h - 10)

TRAIN_DIR = 'dataset/train_data/cursor_data.csv'
TEST_DIR = 'dataset/test_data/cursor_data.csv'
