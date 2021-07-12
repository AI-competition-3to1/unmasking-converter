import os
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from utils.generate import generate_target


class MaskDataset(object):
    def __init__(self, config):
        self.transforms = transforms.Compose([transforms.ToTensor(), ])
        
        basedir = config["directory"]
        self.imgdir = os.path.join(basedir, "images")
        self.anodir = os.path.join(basedir, "annotations")

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


class MaskDataLoader:
    def __init__(self, config, dataset):
        BATCH_SIZE = config["batch_size"]
        
        def collate_fn(batch):
            return tuple(zip(*batch))

        self.loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE, 
            collate_fn=collate_fn
        )
        