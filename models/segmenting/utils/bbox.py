import torch

def scale_bbox(preds):
    boxes = list()
    
    for i, box in enumerate(preds["boxes"]):
        ibox = box_scaler(box)
        boxes.append(ibox)
        
        if i == 0:
            continue

        for j, jbox in enumerate(boxes[:-1]):
            if inside_box(jbox["sbox"], ibox["box"]):
                preds["boxes"] = torch.cat([preds["boxes"][:j], preds["boxes"][j+1:]])
                preds["labels"] = torch.cat([preds["labels"][:j], preds["labels"][j+1:]])
                boxes.pop(j)
                i -= 1
                break
            elif inside_box(ibox["sbox"], jbox["box"]):
                preds["boxes"] = torch.cat([preds["boxes"][:i], preds["boxes"][i+1:]])
                preds["labels"] = torch.cat([preds["labels"][:i], preds["labels"][i+1:]])
                boxes.pop(-1)
                i -= 1
                break

    return preds


def inside_box(s, b):
    return (s[0] >= b[0]) and (s[1] >= b[1]) and (s[2] <= b[2]) and (s[3] <= b[3])


def box_scaler(box):
    xmin, ymin, xmax, ymax = box

    xcenter = (xmin + xmax) / 2
    ycenter = (ymin + ymax) / 2
    width = (xmax - xmin) 
    height = (ymax - ymin) * 1.5
    ycenter -= 0.15 * height
    
    length = width if width > height else height

    xmin = int(xcenter - length / 2)
    xmax = int(xcenter + length / 2)
    ymin = int(ycenter - length / 2)
    ymax = int(ycenter + length / 2)
    
    sxmin = int(xcenter - 0.8 * length / 2)
    sxmax = int(xcenter + 0.8 * length / 2)
    symin = int(ycenter - 0.8 * length / 2)
    symax = int(ycenter + 0.8 * length / 2)

    return {
        "box": (xmin, ymin, xmax, ymax),
        "sbox": (sxmin, symin, sxmax, symax),
    }