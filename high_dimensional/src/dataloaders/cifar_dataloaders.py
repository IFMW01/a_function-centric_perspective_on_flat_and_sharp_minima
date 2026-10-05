import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import tarfile
import io
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DataLoaderManagerCIFAR:
    def __init__(
        self,
        config,
        dataset_name: str,
        seed: int,
        corruption="gaussian_noise",
        corruption_severity=1,
        rand_prob=0.0,
        flip_prob=0.5,
    ):
        self.config = config
        self.corruption = corruption
        self.corruption_severity = corruption_severity
        self.rand_prob = rand_prob
        self.flip_prob = flip_prob
        self.dataset_name = dataset_name
        if self.dataset_name == "CIFAR10":
            self.dataset = datasets.CIFAR10
            self.num_classes = 10
        elif self.dataset_name == "CIFAR100":
            self.dataset = datasets.CIFAR100
            self.num_classes = 100
        elif self.dataset_name == "CIFAR10-C":
            self.dataset = CorruptionCIFAR(
                # Download from https://zenodo.org/records/2535967
                "./CIFAR-10-C.tar",
                self.corruption,
                self.corruption_severity,
            )
            self.num_classes = 10
        elif self.dataset_name == "CIFAR100-C":
            self.dataset = CorruptionCIFAR(
                # Download from https://zenodo.org/records/3555552
                "./CIFAR-100-C.tar",
                self.corruption,
                self.corruption_severity,
            )
            self.num_classes = 100
        elif self.dataset_name == "CIFAR10R":
            self.dataset = datasets.CIFAR10
            self.num_classes = 10
        elif self.dataset_name == "CIFAR100R":
            self.dataset = datasets.CIFAR100
            self.num_classes = 100

        else:
            raise ValueError("Only CIFAR10, CIFAR10-C CIFAR100 or CIFAR100-C supported")

        self.aug_transformations = transforms.Compose(
            [
                transforms.RandomCrop(
                    32, padding=4, fill=128
                ),  # fill parameter needs torchvision installed from source
                transforms.RandomHorizontalFlip(p=self.flip_prob),
                # CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.base_transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.seed = seed

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.dataset_name == "CIFAR10R" or self.dataset_name == "CIFAR100R":
            print("Loading random label train data")
            train_dataset = get_random_cifar_dataset(
                self.dataset,
                self.num_classes,
                corrupt_prob=self.rand_prob,
                root=self.config.data_dir,
                download=True,
                transform=self.aug_transformations
                if self.config.aug
                else self.base_transformations,
                train=True,
            )
            eval_dataset = self.dataset(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=self.base_transformations,
            )
        else:
            print("Loading normal train data")
            if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
                train_dataset = self.dataset(
                    root=self.config.data_dir,
                    train=True,
                    download=True,
                    transform=self.aug_transformations
                    if self.config.aug
                    else self.base_transformations,
                )

                eval_dataset = self.dataset(
                    root=self.config.data_dir,
                    train=False,
                    download=True,
                    transform=self.base_transformations,
                )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            num_workers=12,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return train_dataloader, test_dataloader

    def get_dataloader_aug(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        print("Loading normal train data")
        if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            train_dataset = self.dataset(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.base_transformations,
            )

            eval_dataset = self.dataset(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=self.base_transformations,
            )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            num_workers=12,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return train_dataloader, test_dataloader

    def get_corrupted_dataset(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        corrupt_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=12,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )
        return corrupt_dataloader

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_sharpness_loader(self):
        g = torch.Generator()
        g.manual_seed(42)

        if self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            train_dataset = self.dataset(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.aug_transformations
                if self.config.aug
                else self.base_transformations,
            )

        sharp_dataset, dummmy_dataset = torch.utils.data.random_split(
            dataset=train_dataset,
            lengths=[int((len(train_dataset) * 0.2)), int((len(train_dataset) * 0.8))],
            generator=g,
        )

        sharp_dataloader = torch.utils.data.DataLoader(
            sharp_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return sharp_dataloader

    def get_constant_sharpness_loader(self):
        g = torch.Generator()
        if self.dataset_name == "CIFAR10R" or self.dataset_name == "CIFAR100R":
            train_dataset = self.dataset(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.aug_transformations
                if self.config.aug
                else self.base_transformations,
            )

        sharp_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return sharp_dataloader

    def get_constant_sharpness_loader_train_test(self):
        g = torch.Generator()
        if self.dataset_name == "CIFAR10R" or self.dataset_name == "CIFAR100R":
            train_dataset = self.dataset(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.aug_transformations
                if self.config.aug
                else self.base_transformations,
            )

            eval_dataset = self.dataset(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=self.base_transformations,
            )

        sharp_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=12,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        test_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            num_workers=12,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        return sharp_dataloader, test_dataloader


# Reused from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py


class CIFARRandomLabelsBase:
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    num_classes: int
      The number of classes in the dataset.
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    """

    def __init__(self, num_classes, corrupt_prob=0.0, **kwargs):
        super(CIFARRandomLabelsBase, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


class CorruptionCIFAR(datasets.VisionDataset):
    def __init__(self, root: str, name: str, corruption_severity: int, transform=True):
        corruptions = [
            "fog",
            "jpeg_compression",
            "zoom_blur",
            "speckle_noise",
            "glass_blur",
            "spatter",
            "shot_noise",
            "defocus_blur",
            "elastic_transform",
            "gaussian_blur",
            "frost",
            "saturate",
            "brightness",
            "snow",
            "gaussian_noise",
            "motion_blur",
            "contrast",
            "impulse_noise",
            "pixelate",
        ]
        assert name in corruptions
        super(CorruptionCIFAR, self).__init__(
            tarfile.open(name=root, mode="r", fileobj=None, bufsize=10240),
            transform=transform,
        )
        corruption = 10000 * corruption_severity
        if transform:
            self.base_transformations = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        dataset = root.split("/")[-1]
        with tarfile.open(root, "r:") as tar:
            if dataset == "CIFAR-100-C.tar":
                file_obj = tar.extractfile(f"CIFAR-100-C/{name}.npy")
                labels_obj = tar.extractfile(f"CIFAR-100-C/labels.npy")
            elif dataset == "CIFAR-10-C.tar":
                file_obj = tar.extractfile(f"CIFAR-10-C/{name}.npy")
                labels_obj = tar.extractfile(f"CIFAR-10-C/labels.npy")
            if file_obj:
                self.data = np.load(io.BytesIO(file_obj.read()))
                self.data = self.data[(corruption - 10000) : corruption]
            if labels_obj:
                self.targets = np.load(io.BytesIO(labels_obj.read()))
                self.targets = self.targets[corruption - 10000 : corruption]

    def __getitem__(self, index):
        if self.transform == True:
            img = self.base_transformations(self.data[index])
        img, targets = img, torch.from_numpy(np.array(self.targets[index]))
        return img, targets

    def __len__(self):
        return len(self.data)


def get_random_cifar_dataset(dataset, num_classes, corrupt_prob=0.0, **kwargs):

    class CIFARRandomLabels(CIFARRandomLabelsBase, dataset):
        pass

    return CIFARRandomLabels(num_classes, corrupt_prob, **kwargs)
