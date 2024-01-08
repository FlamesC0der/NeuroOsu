import csv
import numpy as np
import base64
import os


def load_data(path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        data = [{'img': d['img'], 'pos': (float(d['x']), float(d['y']))} for d in list(reader)]
    return data


def save_data(path: str, img: np.ndarray, x: float, y: float, w: int, h: int) -> None:
    write_header = not os.path.exists(path) or os.stat(path).st_size == 0
    with open(path, 'a') as f:
        fieldnames = ['img', 'x', 'y']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        print("save", x, y, x / w, y / h)

        if write_header:
            writer.writeheader()
        writer.writerow({'img': base64.b64encode(img.tobytes()).decode('utf-8'), 'x': x / w, 'y': y / h})
