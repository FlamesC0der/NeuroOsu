import csv
import numpy as np
import base64


def load_data(path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        data = [{'img': d['img'], 'pos': (float(d['x']), float(d['y']))} for d in list(reader)]
    return data


def save_data(img: np.ndarray, x: float, y: float, w: int, h: int) -> None:
    with open("dataset/cursor_data.csv", 'a') as f:
        fieldnames = ['img', 'x', 'y']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        print("save")
        # if len(writer) == 0:
        #     writer.writeheader()
        writer.writerow({'img': base64.b64encode(img.tobytes()).decode('utf-8'), 'x': x / w, 'y': y / h})
