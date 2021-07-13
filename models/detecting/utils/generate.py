import os
import torch
from bs4 import BeautifulSoup


def generate_box(obj):
    xmin = int(obj.find("xmin").text)
    ymin = int(obj.find("ymin").text)
    xmax = int(obj.find("xmax").text)
    ymax = int(obj.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find("name").text == "with_mask":
        return 1
    elif obj.find("name").text == "mask_weared_incorrect":
        return 2
    return 0


def generate_target(image_id, file):
    if not os.path.exists(file):
        target = {}
        target["boxes"] = None
        target["labels"] = None
        target["image_id"] = None
        return target

    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "xml")
        objects = soup.find_all("object")

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img_id = torch.tensor([image_id])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        return target
