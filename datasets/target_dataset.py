
import os
import torch
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def open_image(path):
    return Image.open(path).convert("RGB")

class TargetDataset(data.Dataset):
    def __init__(self, dataset_folder):

        super().__init__()
        self.dataset_folder = dataset_folder
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.images_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))
        

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        label = torch.tensor(1)
        return normalized_img, label
    
    def __len__(self):
        return len(self.images_paths)

class DomainAdaptationDataLoader(data.DataLoader):
    def __init__(self, source_dataset, target_dataset, pseudo_dataset=None, pseudo=True, *args, **kwargs):
        if pseudo and pseudo_dataset:
          self.pseudo_dim = int(kwargs["batch_size"] * 1 / 3)
        self.source_dim = int(kwargs["batch_size"] * 1 / 3)
        self.target_dim = kwargs["batch_size"] - self.source_dim - self.pseudo_dim
        del kwargs["batch_size"]
        self.source_domain_loader = data.DataLoader(source_dataset, batch_size=self.source_dim, **kwargs)
        self.source_domain_iterator = self.source_domain_loader.__iter__()
        self.target_domain_loader = data.DataLoader(target_dataset, batch_size=self.target_dim, **kwargs)
        self.target_domain_iterator = self.target_domain_loader.__iter__()
        if pseudo and pseudo_dataset:
          self.pseudo_domain_loader = data.DataLoader(pseudo_dataset, batch_size=self.source_dim, **kwargs)
          self.pseudo_domain_iterator = self.pseudo_domain_loader.__iter__()
    def __iter__(self):
        return self

    def __next__(self):
        try:
            source_data,_,source_labels = next(self.source_domain_iterator)
        except StopIteration:
            self.source_domain_iterator = self.source_domain_loader.__iter__()
            source_data,_,source_labels = next(self.source_domain_iterator)

        try:
            target_data, target_labels = next(self.target_domain_iterator)
        except StopIteration:
            self.target_domain_iterator = self.target_domain_loader.__iter__()
            target_data, target_labels  = next(self.target_domain_iterator)

        if hasattr(self, 'pseudo_domain_loader'):
          try:
                pseudo_data,_ ,pseudo_labels= next(self.pseudo_domain_iterator)
          except StopIteration:
                self.pseudo_domain_iterator = self.pseudo_domain_loader.__iter__()
                pseudo_data,_, pseudo_labels = next(self.pseudo_domain_iterator)
          batch = (torch.cat((source_data,pseudo_data,target_data),0),torch.cat((source_labels, pseudo_labels, target_labels),0))
        else:  
          batch = (torch.cat((source_data, target_data),0),torch.cat((source_labels, target_labels),0))
        return batch