import os
from PIL import Image
from utils.generate import generate_target


class MaskDataset(object):
    def __init__(self, transforms, datadir):
        self.transforms = transforms
        self.datadir = datadir
        self.imgdir = os.path.join(datadir, "images")
        self.anodir = os.path.join(datadir, "annotations")

        self.imgs = list(sorted(os.listdir(self.imgdir)))

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss'+ str(idx) + '.png'
        file_label = 'maksssksksss'+ str(idx) + '.xml'
        img_path = os.path.join(self.imgdir, file_image)
        label_path = os.path.join(self.anodir, file_label)
        img = Image.open(img_path).convert("RGB")

        #Generate Label
        target = generate_target(idx, label_path)
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
    