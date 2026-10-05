import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import numpy as np
import tarfile
import io
import os
import torchvision.transforms.v2 as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image


def covert2rgb(dataset):
    dataset["image"] = [
        image if image.mode == "RGB" else image.convert("RGB")
        for image in dataset["image"]
    ]
    return dataset


def covert2rgbcorrupt(dataset):
    dataset = [
        image if image.mode == "RGB" else image.convert("RGB") for image in dataset
    ]
    return dataset


class TinyImagenetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = dataset.map(covert2rgb, batched=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]["image"]
        target = self.data[idx]["label"]
        if self.transform:
            sample = self.transform(sample)

        return sample, target


class DataLoaderManagerImageNet:
    def __init__(
        self,
        config,
        dataset_name: str,
        seed: int,
        corruption="gaussian_noise",
        corruption_severity=1,
    ):
        self.config = config

        self.dataset_name = dataset_name
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        if dataset_name == "TinyImageNet":
            self.num_classes = 200
        elif dataset_name == "ImageNet1K":
            self.num_classes = 1000
        elif self.dataset_name == "TinyImageNet-C":
            self.dataset = CorruptionTinyImageNet(
                # Download from TODO
                "./Tiny-ImageNet-C.tar",
                self.corruption,
                self.corruption_severity,
            )
            self.num_classes = 200
        else:
            raise ValueError("Only TinyImageNet and ImageNet1K supported")

        self.aug_transformations = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.base_transformations = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.seed = seed

    def apply_transform_aug(self, example):
        example["image"] = self.aug_transformations(example["image"])
        return example

    def apply_transform(self, example):
        example["image"] = self.base_transformations(example["image"])
        return example

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        print("Loading normal train data")
        if self.dataset_name == "TinyImageNet":
            dataset = load_dataset("zh-plus/tiny-imagenet")
        if self.config.aug:
            train_dataset = TinyImagenetDataset(
                dataset["train"], transform=self.aug_transformations
            )
            test_dataset = TinyImagenetDataset(
                dataset["valid"], transform=self.aug_transformations
            )
        else:
            train_dataset = TinyImagenetDataset(
                dataset["train"], transform=self.base_transformations
            )
            test_dataset = TinyImagenetDataset(
                dataset["valid"], transform=self.base_transformations
            )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return train_dataloader, test_dataloader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_sharpness_loader(self):
        g = torch.Generator()
        g.manual_seed(42)

        print("Loading normal train data")
        if self.dataset_name == "TinyImageNet":
            dataset = load_dataset("zh-plus/tiny-imagenet")
        if self.config.aug:
            train_dataset = TinyImagenetDataset(
                dataset["train"], transform=self.aug_transformations
            )
        else:
            train_dataset = TinyImagenetDataset(
                dataset["train"], transform=self.base_transformations
            )

        sharp_dataset, dummmy_dataset = torch.utils.data.random_split(
            dataset=train_dataset,
            lengths=[int((len(train_dataset) * 0.5)), int((len(train_dataset) * 0.5))],
            generator=g,
        )

        sharp_dataloader = torch.utils.data.DataLoader(
            sharp_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return sharp_dataloader

    def get_corrupted_dataset(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        corrupt_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )
        return corrupt_dataloader

    def get_dataloader_aug(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.dataset_name == "TinyImageNet":
            dataset = load_dataset("zh-plus/tiny-imagenet")
            train_dataset = TinyImagenetDataset(
                dataset["train"], transform=self.base_transformations
            )
            test_dataset = TinyImagenetDataset(
                dataset["valid"], transform=self.base_transformations
            )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=4,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return train_dataloader, test_dataloader


class CorruptionTinyImageNet(datasets.VisionDataset):
    CORRUPTIONS = [
        "shot_noise",
        "brightness",
        "pixelate",
        "glass_blur",
        "motion_blur",
        "impulse_noise",
        "frost",
        "jpeg_compression",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "snow",
        "fog",
        "gaussian_noise",
        "zoom_blur",
    ]

    def __init__(
        self,
        root: str,
        name: str,
        corruption_severity: int,
        transform: bool = True,
        tar_path: str = "./tiny-imagenet-c.tar",
    ):
        super().__init__(root)

        assert name in self.CORRUPTIONS, (
            f"Unknown corruption '{name}'. Choose from: {self.CORRUPTIONS}"
        )
        assert 1 <= corruption_severity <= 5, (
            f"Severity must be between 1 and 5, got {corruption_severity}"
        )

        self.name = name
        self.corruption_severity = corruption_severity
        self.base_transformations = None

        # Load HuggingFace label ordering as ground truth
        hf_dataset = load_dataset("zh-plus/tiny-imagenet")
        hf_class_names = hf_dataset["train"].features["label"].names

        self.classes = list(hf_class_names)
        wnid_to_index = {wnid: idx for idx, wnid in enumerate(hf_class_names)}

        if transform:
            self.base_transformations = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        print(f"Loading corruption='{name}', severity={corruption_severity} ...")

        img_arr, label_arr = [], []

        tar_paths = [p for p in [tar_path] if os.path.exists(p)]
        if len(tar_paths) == 0:
            raise RuntimeError("No tar files found.")
        if len(tar_paths) < 2:
            print(f"Warning: only found {tar_paths}, missing classes may be absent.")

        for tp in tar_paths:
            print(f"Reading {tp} ...")
            with tarfile.open(tp, "r") as tar:
                members = [
                    m
                    for m in tar.getmembers()
                    if m.isfile() and m.name.upper().endswith(".JPEG")
                ]

                for member in members:
                    parts = member.name.split("/")

                    if len(parts) != 5:
                        continue

                    _, corruption, severity_str, wnid, _ = parts

                    if corruption != name:
                        continue
                    if severity_str != str(corruption_severity):
                        continue
                    if wnid not in wnid_to_index:
                        continue

                    label = wnid_to_index[wnid]

                    img_obj = tar.extractfile(member)
                    if img_obj is None:
                        continue

                    try:
                        image = Image.open(io.BytesIO(img_obj.read())).convert("RGB")
                        img_arr.append(image)
                        label_arr.append(label)
                    except Exception as e:
                        print(f"Warning: could not load {member.name}: {e}")
                        continue

        if len(img_arr) == 0:
            raise RuntimeError(
                f"No images found for corruption='{name}', severity={corruption_severity}. "
                f"Check tar structure and parameters."
            )

        loaded_classes = len(set(label_arr))
        print(f"Loaded {len(img_arr)} images across {loaded_classes} classes.")
        if loaded_classes < 200:
            print(f"Warning: only {loaded_classes}/200 classes loaded.")

        self.data = img_arr
        self.targets = np.asarray(label_arr, dtype=int)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx]

        if self.base_transformations is not None:
            img = self.base_transformations(img)

        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.data)
