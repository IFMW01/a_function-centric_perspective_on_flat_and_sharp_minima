import string
import numpy as np
import math
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import torch
from torch.utils.data import Dataset, DataLoader

allowed_characters = string.ascii_letters + "' "

stoi = {
    "[PAD]": 0,
    "[UNK]": 1,
}
stoi.update({ch: i + 2 for i, ch in enumerate(allowed_characters)})
itos = {
    "[PAD]": 0,
    "[UNK]": 1,
}
itos.update({i + 2: ch for i, ch in enumerate(allowed_characters)})


def encode(s):
    encoding = []
    for c in s:
        if c in stoi:
            encoding.append(stoi[c])
        else:
            encoding.append(1)
    return encoding


def decode(l):
    return "".join([itos[i] for i in l])


class DataLoaderManagerNAMES:
    def __init__(self, config, dataset_name: str, seed: int, device):
        self.config = config
        self.dataset_name = dataset_name
        self.seed = seed
        self.device = device
        self.num_classes = 15

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        dataset = torch.load(
            "./src/simulated_name_dataset.pth",
            weights_only=False,
        )

        if self.config.aug:
            augmentor = nac.keyboard.KeyboardAug(
                aug_char_p=0.1,
                include_special_char=False,
                include_numeric=False,
                aug_char_min=1,
            )
            augmented_dataset = NamesAugDataset(
                **dataset["train"],
                encoder=encode,
                max_length=25,
                device=self.device,
                augmentor=augmentor,
            )
            train_dataloader = DataLoader(
                augmented_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=g,
            )
        else:
            train_dataset = NamesDataset(
                **dataset["train"], encoder=encode, max_length=25, device=self.device
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=g,
            )

        test_dataset = NamesDataset(
            **dataset["test"], encoder=encode, max_length=25, device=self.device
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False, generator=g
        )

        return train_dataloader, test_dataloader

    def get_sharpness_loader(self):
        g = torch.Generator()
        g.manual_seed(42)
        dataset = torch.load("./src/simulated_name_dataset.pth", weights_only=False)

        train_dataset = NamesDataset(
            **dataset["train"], encoder=encode, max_length=25, device=self.device
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, generator=g
        )
        return train_dataloader


class NamesDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        encoder,
        pad_id=0,
        max_length=25,
        device=torch.get_default_device(),
    ):
        self.max_length = max_length
        padded_inputs = []
        attention_masks = []
        for seq in data:
            seq = encoder(seq)
            num_pads = self.max_length - len(seq)

            # Right-pad sequences to reach max_len
            padded_inputs.append([pad_id] * num_pads + seq)

            # 1 for valid tokens, 0 for padded tokens
            mask = [pad_id] * num_pads + [1] * len(seq)
            attention_masks.append(mask)
        self.data = torch.tensor(padded_inputs, device=device)
        self.attention_masks = torch.tensor(attention_masks, device=device)
        self.labels = torch.tensor(labels, device=device, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        mask = self.attention_masks[idx]
        label = self.labels[idx]
        return {"input_ids": datum, "attention_mask": mask, "labels": label}


class NamesAugDataset(Dataset):
    def __init__(
        self,
        data,
        labels,
        encoder,
        augmentor,
        pad_id=0,
        max_length=25,
        device=torch.get_default_device(),
    ):
        self.max_length = max_length
        self.augmentor = augmentor
        self.encoder = encoder
        self.pad_id = pad_id
        self.data = np.array(data)
        self.labels = torch.tensor(labels, device=device, dtype=torch.long)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = list([idx])
        padded_inputs = []
        attention_masks = []
        for seq in self.data[idx]:
            seq = str(seq)
            seq = self.augmentor.augment(seq)[0]
            seq = self.encoder(seq)
            num_pads = self.max_length - len(seq)

            # Right-pad sequences to reach max_len
            padded_inputs.append([self.pad_id] * num_pads + seq)

            # 1 for valid tokens, 0 for padded tokens
            mask = [self.pad_id] * num_pads + [1] * len(seq)
            attention_masks.append(mask)
        label = self.labels[idx]
        return {
            "input_ids": torch.tensor(*padded_inputs, device=self.device),
            "attention_mask": torch.tensor(*attention_masks, device=self.device),
            "labels": torch.tensor(label.item(), device=self.device),
        }
