import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class Cityscaples(Dataset):
    def __init__(self, root='/data/cityscapes', split='train', image_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.label_transform = label_transform

        # Load images and labels
        self.images = []
        self.labels = []
        image_dir = os.path.join(root, 'leftImg8bit', split)
        label_dir = os.path.join(root, 'gtFine', split)
        
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            for image_file in os.listdir(city_image_dir):
                if image_file.endswith('.png'):
                    self.images.append(os.path.join(city_image_dir, image_file))
                    label_file = image_file.replace('leftImg8bit', 'gtFine_labelIds')
                    self.labels.append(os.path.join(city_label_dir, label_file))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index])

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        # Convert label to numpy array and then to tensor
        label = torch.from_numpy(np.array(label, dtype=np.int64)).long().squeeze(0) # .squeeze(0) is used to remove the channel dimension
        
        return image, label
