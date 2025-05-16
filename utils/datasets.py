import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import pathlib
import pyarrow.parquet as pq
import pandas as pd
from torchvision.datasets import ImageFolder, DatasetFolder

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


class ParquetDataset(Dataset):

    def __init__(self, parquet_path, map_path, client_idx=None, transform=None, target_transform=None):

        self.parquet_path = parquet_path
        self.map_path = map_path
        self.transform = transform
        self.target_transform = target_transform
        self.client_idx = client_idx

        self.table = pq.read_table(parquet_path)
        self.df = self.table.to_pandas()

        self.client_files = set()
        if client_idx is not None:
            try:
                with open(map_path, 'r') as f:
                    client_data = json.load(f)
                client_key = f"client_{client_idx}"
                if client_key in client_data["clients"]:
                    client_labels = client_data["clients"][client_key]["labels"]
                    for label_data in client_labels.values():
                        for item in label_data:
                            self.client_files.add(item["filename"])
                    if 'filename' in self.df.columns:
                        self.df = self.df[self.df['filename'].isin(self.client_files)]
                    else:
                        raise ValueError("Parquet file lacks filename column")
            except Exception as e:
                print(client_idx)
                print(f"Processing map file failed: {e}")

        elif map_path is not None:
            try:
                with open(map_path, 'r') as f:
                    client_data = json.load(f)

                if "labels" in client_data:
                    for label_data in client_data["labels"].values():
                        for item in label_data:
                            self.client_files.add(item["filename"])

                    if 'filename' in self.df.columns:
                        self.df = self.df[self.df['filename'].isin(self.client_files)]
                    else:
                        raise ValueError("Parquet file lacks filename column")
                else:
                    raise ValueError("Map file does not contain 'labels' key")


            except Exception as e:
                print(client_idx)
                print(f"Processing map file failed: {e}")

        required_columns = ['image_data', 'class_id', 'filename']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Parquet lacks column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_bytes = self.df.iloc[idx]['image_data']
        filename = self.df.iloc[idx]['filename']
        label = int(self.df.iloc[idx]['class_id'])

        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"解码失败: {filename} ({e})")
            return None

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class ImageFolder_SingleDatasetTIF(DatasetFolder):
    def __init__(self, root, client_idx=None, train=True, dataidxs=None, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.client_idx = client_idx
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if client_idx is not None:
            client_root = f"{root}/client_{client_idx}"
        else:
            client_root = root

        print(client_root)

        if not os.path.exists(client_root):
            logger.error(f"Path {client_root} does not exist.")
            raise ValueError(f"Path {client_root} does not exist.")

        if not os.path.isdir(client_root):
            logger.error(f"Path {client_root} is not a valid directory.")
            raise ValueError(f"Path {client_root} is not a valid directory.")

        # Custom class-to-index mapping
        self.label_mapping = {
            "Agriculture": 0,
            "Bareland": 1,
            "Forest": 2,
            "Residential": 3,
            "River": 4,
            "Airport": 5,
            "Beach": 6,
            "Highway": 7,
            "Industrial": 8,
            "Port": 9,
            "Overpass": 10,
            "Parkinglot": 11,
            "Bridge": 12,
            "Mountain": 13,
            "Meadow": 14
        }

        # Use the custom find_classes method
        classes, class_to_idx = self.find_classes(client_root)

        imagefolder_obj = ImageFolder(client_root, transform=self.transform, target_transform=self.target_transform,allow_empty=True)

        # Update class mapping
        imagefolder_obj.class_to_idx = class_to_idx

        # Modify samples to use custom labels
        self.samples = [
            (path, class_to_idx[classes[original_label]])
            for path, original_label in imagefolder_obj.samples
        ]

        # Store updated information
        self.loader = imagefolder_obj.loader
        self.class_to_idx = class_to_idx
        self.classes = classes

    def find_classes(self, dir):
        # Get all class directories
        classes = sorted([d.name for d in os.scandir(dir) if d.is_dir()])
        # Create class-to-index mapping (only for classes in label_mapping)
        class_to_idx = {cls_name: self.label_mapping[cls_name]
                        for cls_name in classes if cls_name in self.label_mapping}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]

        if target == -1:
            logger.warning(f"Unknown label in sample {path}, skipping")
            return None

        sample = self.loader(path)

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
