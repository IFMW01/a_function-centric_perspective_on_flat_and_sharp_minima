import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.optim as optim
import argparse
import copy
from copy import deepcopy
import os
import random
from pathlib import Path
import json
import io
import numpy as np
import dataloaders.cifar_dataloaders as CIFAR_dataloader
import dataloaders.imagenet_dataloaders as IMAGENET_dataloader
import dataloaders.names_dataloader as NAMES_dataloader
import trainer.trainer as Trainer
import trainer.hf_trainer as hf_Trainer
from models.vgg import make_vgg
from models.resnet import ResNet18, ResNet50
import models.vit as vit_model
import utils.model_metrics as mm
import utils.sharpness_metrics as sm
import utils.sharpness_metrics_llm as lsm
from models.custom_pretrained import ResNet18PT, VGG19BNPT
from transformers import GPT2Config, GPT2Model, GPT2ForSequenceClassification
from sam import SAM


def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")

    # Required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Datraset to train on (CIFAR10, CIFAR100 or TinyImageNet)",
    )

    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="VGG, ResNet or ViT ",
    )

    parser.add_argument("--seed", required=True, type=int, help="Seed to train on")

    parser.add_argument(
        "--num_epochs", required=True, type=int, help="Number of epochs to train for."
    )

    parser.add_argument("--save_name", required=True, type=str, help="Project Name")

    parser.add_argument(
        "--models_dir",
        required=False,
        type=str,
        default="./models",
        help="Directory to save models to.",
    )

    parser.add_argument(
        "--data_dir",
        required=False,
        type=str,
        default="./data/cifar",
        help="Data directory (Change when not using CIFAR)",
    )

    # Optional arguments
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=256,
        help="Batch size to train on",
    )

    parser.add_argument(
        "--learning_rate",
        required=False,
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--optimizer",
        required=False,
        type=str,
        default="SGD",
        help="Optimizer (SGD,Adam)",
    )

    parser.add_argument(
        "--momentum", required=False, type=int, default=0.9, help="Value for momentum"
    )

    parser.add_argument(
        "--criterion",
        required=False,
        type=str,
        default="Cross-entropy",
        help="Loss metric",
    )

    parser.add_argument(
        "--dropout", required=False, type=float, default=0.0, help="Dropout  value"
    )

    parser.add_argument(
        "--weight_decay",
        required=False,
        type=float,
        default=0.0,
        help="Weight decay value (suggested 1e-4)",
    )

    parser.add_argument(
        "--SAM", required=False, type=bool, default=False, help="Use SAM as optimizer"
    )

    parser.add_argument(
        "--rho", required=False, type=float, default=0.05, help="Use SAM as optimizer"
    )

    parser.add_argument(
        "--aug",
        required=False,
        type=bool,
        default=False,
        help="Augmentation (True/False)",
    )

    parser.add_argument(
        "--scheduler",
        required=False,
        type=bool,
        default=False,
        help="Use Cosine Scheduler (True/False)",
    )

    parser.add_argument(
        "--sharpness",
        required=False,
        type=bool,
        default=False,
        help="Measuring flatness/sharpness",
    )

    parser.add_argument(
        "--innit",
        required=False,
        type=bool,
        default=False,
        help="Sharpness for innit",
    )

    parser.add_argument(
        "--corrupt", required=False, type=bool, default=False, help="Robustness Test"
    )

    parser.add_argument(
        "--rand_prob",
        required=False,
        type=float,
        default=0.0,
        help="Random Labels Probability",
    )

    args = parser.parse_args()
    return args


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_file_name, save_dir):

    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    full_save_dir = save_dir / f"{save_file_name}.pth"
    torch.save(model.state_dict(), full_save_dir)
    print("-----------------")
    print(f"Model saved at: {full_save_dir}")
    print("-----------------")


def instantiate_model(args, data_loader_manager, device):
    scheduler = False
    if args.model_name.startswith("VGG") and args.model_name != "VGG19BNPT":
        model = make_vgg(
            args.model_name, data_loader_manager.num_classes, args.dropout
        ).to(device)
    elif args.model_name == "ResNet18":
        model = ResNet18(data_loader_manager.num_classes).to(device)
    elif args.model_name == "ResNet50":
        model = ResNet50(data_loader_manager.num_classes).to(device)
    elif args.model_name == "ResNet18PT":
        model = ResNet18PT(num_classes=data_loader_manager.num_classes)
        model.to(device)
    elif args.model_name == "VGG19BNPT":
        model = VGG19BNPT(num_classes=data_loader_manager.num_classes)
        model.to(device)
    elif args.model_name == ("ViT"):
        model = vit_model.ViT(
            image_size=32,
            patch_size=4,
            num_classes=data_loader_manager.num_classes,
            dim=512,
            depth=6,
            heads=6,
            mlp_dim=256,
            dropout=args.dropout,
            emb_dropout=args.dropout,
        ).to(device)
    elif args.model_name == "GPT2":
        model_config = {
            "n_embd": 128,
            "n_layer": 4,
            "n_head": 4,
            "n_positions": 25,
            "vocab_size": 56,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "bos_token_id": None,
            "eos_token_id": None,
            "pad_token_id": 0,
            "num_labels": 15,
            "problem_type": "single_label_classification",
            "output_hidden_states": True,
        }
        configuration = GPT2Config(**model_config)
        model = GPT2ForSequenceClassification(configuration).to(device)
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")

    learning_rate = args.learning_rate
    momentum = args.momentum

    if "Cross-entropy" in args.criterion:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Only Cross-entropy and MSE are supported")

    if "SGD" == args.optimizer:
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=args.weight_decay,
        )
    elif "Adam" == args.optimizer:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Only SGD and Adam are supported")
    return model, criterion, optimizer, scheduler


def train_model(args, data_loader_manager, device):
    print(f"Loading: {args.dataset}:")

    if (
        args.dataset == "CIFAR10"
        or args.dataset == "CIFAR10R"
        or args.dataset == "CIFAR100"
        or args.dataset == "CIFAR100R"
        or args.dataset == "TinyImageNet"
        or args.dataset == "NAMES"
    ):
        train_dataloader, test_dataloader = data_loader_manager.get_dataloaders()
    if args.sharpness:
        mertics = [
            "acc",
            "ece",
            "squared_euclidean_norm",
            "fisher_rao_norm",
            "sam_sharpness",
            "relative_flatness",
        ]
    if args.dataset == "TinyImageNet" or "CIFAR100":
        sharpness_dataloader = data_loader_manager.get_sharpness_loader()
        if args.dataset == "TinyImageNet":
            mertics = [
                "acc",
                "ece",
                "squared_euclidean_norm",
                "fisher_rao_norm",
                "sam_sharpness",
            ]
    if args.dataset == "CIFAR10R" or args.dataset == "CIFAR100R":
        print("Locally Constant Lables for Sharpness")
        sharpness_dataloader = data_loader_manager.get_constant_sharpness_loader()
    initialisation_path_dir = f"./models/{args.dataset}/{args.model_name}/{args.seed}/"
    initialisation_path = f"{initialisation_path_dir}initialisation.pth"
    model, criterion, optimizer, scheduler = instantiate_model(
        args, data_loader_manager, device
    )

    sharpness_path = (
        f"./models/{args.dataset}/{args.model_name}/{args.seed}/sharpness.json"
    )
    if os.path.exists(initialisation_path_dir):
        innit_state_dict = torch.load(initialisation_path)
        model.load_state_dict(innit_state_dict)
        if args.sharpness:
            if os.path.exists(sharpness_path):
                with open(sharpness_path) as f:
                    sharpness_dict = json.load(f)
                for metric in mertics:
                    if metric not in sharpness_dict["model_innit"]:
                        innit_model = deepcopy(model)
                        if (
                            (
                                args.dataset == "TinyImageNet"
                                and metric == "sam_sharpness"
                            )
                            or (
                                args.dataset == "TinyImageNet"
                                and metric == "relative_flatness"
                            )
                            or (
                                args.dataset == "CIFAR100"
                                and metric == "relative_flatness"
                            )
                        ):
                            Metrics_Processor = sm.MetricsProcessor(
                                [metric],
                                innit_model,
                                sharpness_dataloader,
                                test_dataloader,
                                device,
                                args.seed,
                                args.model_name,
                                data_loader_manager.num_classes,
                            )
                        else:
                            if args.dataset == "NAMES":
                                Metrics_Processor = lsm.MetricsProcessor(
                                    [metric],
                                    innit_model,
                                    train_dataloader,
                                    test_dataloader,
                                    device,
                                    args.seed,
                                    args.model_name,
                                    data_loader_manager.num_classes,
                                )
                            else:
                                Metrics_Processor = sm.MetricsProcessor(
                                    [metric],
                                    innit_model,
                                    train_dataloader,
                                    test_dataloader,
                                    device,
                                    args.seed,
                                    args.model_name,
                                    data_loader_manager.num_classes,
                                )
                        sharpness_dict["model_innit"][f"{metric}"] = (
                            Metrics_Processor.compute_metrics()[f"{metric}"]
                        )
                        with open(sharpness_path, "w") as f:
                            json.dump(sharpness_dict, f)
    else:
        os.makedirs(initialisation_path_dir, exist_ok=True)
        torch.save(model.state_dict(), initialisation_path)
        if args.sharpness:
            sharpness_dict = {}
            sharpness_dict["model_innit"] = {}
            for metric in mertics:
                innit_model = deepcopy(model)
                if (
                    (args.dataset == "TinyImageNet" and metric == "sam_sharpness")
                    or (
                        args.dataset == "TinyImageNet" and metric == "relative_flatness"
                    )
                    or (args.dataset == "CIFAR100" and metric == "relative_flatness")
                ):
                    Metrics_Processor = sm.MetricsProcessor(
                        [metric],
                        innit_model,
                        sharpness_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )
                else:
                    if args.dataset == "NAMES":
                        Metrics_Processor = lsm.MetricsProcessor(
                            [metric],
                            innit_model,
                            train_dataloader,
                            test_dataloader,
                            device,
                            args.seed,
                            args.model_name,
                            data_loader_manager.num_classes,
                        )
                    else:
                        Metrics_Processor = sm.MetricsProcessor(
                            [metric],
                            innit_model,
                            train_dataloader,
                            test_dataloader,
                            device,
                            args.seed,
                            args.model_name,
                            data_loader_manager.num_classes,
                        )
                sharpness_dict["model_innit"][f"{metric}"] = (
                    Metrics_Processor.compute_metrics()[f"{metric}"]
                )
                with open(sharpness_path, "w") as f:
                    json.dump(sharpness_dict, f)
    base_optimiser = False
    scheduler = False
    if args.SAM:
        if args.scheduler:
            base_optimiser = optimizer
            optimizer = SAM(model.parameters(), base_optimiser, rho=args.rho)
        else:
            optimizer = SAM(model.parameters(), optimizer, rho=args.rho)
    if args.scheduler:
        if args.SAM:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_optimiser, T_max=args.num_epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs
            )
    if args.dataset == "NAMES":
        trainer = hf_Trainer.Trainer(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_epoch=args.num_epochs,
            n_classes=data_loader_manager.num_classes,
            SAM=args.SAM,
            scheduler=scheduler,
            base_optimiser=base_optimiser,
        )
    else:
        print("here")
        trainer = Trainer.Trainer(
            model=model,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            n_epoch=args.num_epochs,
            n_classes=data_loader_manager.num_classes,
            SAM=args.SAM,
            scheduler=scheduler,
            base_optimiser=base_optimiser,
        )

    training_sequence, model = trainer.train()
    save_model(model, args.save_name, args.models_dir)
    with open(f"{args.models_dir}/{args.save_name}.json", "w") as f:
        json.dump(training_sequence, f)

    if args.sharpness:
        eot_sharpness_path = f"{args.models_dir}/{args.save_name}_sharpness.json"
        sharpness_dict = {}
        sharpness_dict["model_eot"] = {}
        for metric in mertics:
            eot_model = deepcopy(model)
            if (
                (args.dataset == "TinyImageNet" and metric == "sam_sharpness")
                or (args.dataset == "TinyImageNet" and metric == "relative_flatness")
                or (args.dataset == "CIFAR100" and metric == "relative_flatness")
            ):
                Metrics_Processor = sm.MetricsProcessor(
                    [metric],
                    eot_model,
                    sharpness_dataloader,
                    test_dataloader,
                    device,
                    args.seed,
                    args.model_name,
                    data_loader_manager.num_classes,
                )
            elif args.dataset == "CIFAR10R" or args.dataset == "CIFAR100R":
                Metrics_Processor = sm.MetricsProcessor(
                    [metric],
                    eot_model,
                    sharpness_dataloader,
                    test_dataloader,
                    device,
                    args.seed,
                    args.model_name,
                    data_loader_manager.num_classes,
                )
            else:
                if args.dataset == "NAMES":
                    Metrics_Processor = lsm.MetricsProcessor(
                        [metric],
                        eot_model,
                        train_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )
                else:
                    Metrics_Processor = sm.MetricsProcessor(
                        [metric],
                        eot_model,
                        train_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )

            sharpness_dict["model_eot"][f"{metric}"] = (
                Metrics_Processor.compute_metrics()[f"{metric}"]
            )
            with open(eot_sharpness_path, "w") as f:
                json.dump(sharpness_dict, f)


def get_data(args, data_loader_manager):
    if (
        args.dataset == "CIFAR10"
        or args.dataset == "CIFAR100"
        or args.dataset == "TinyImageNet"
        or args.dataset == "NAMES"
    ):
        train_dataloader, test_dataloader = data_loader_manager.get_dataloaders()
    elif args.dataset == "CIFAR10R" or args.dataset == "CIFAR100R":
        print("Locally Constant Lables for Sharpness")
        train_dataloader, test_dataloader = (
            data_loader_manager.get_constant_sharpness_loader_train_test()
        )
    return train_dataloader, test_dataloader


def calculate_sharpness(args, data_loader_manager, model_path, sharpness_path, device):
    model, criterion, optimizer, scheduler = instantiate_model(
        args, data_loader_manager, device
    )
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    train_dataloader, test_dataloader = get_data(args, data_loader_manager)

    mertics = [
        "acc",
        "ece",
        "squared_euclidean_norm",
        "fisher_rao_norm",
        "sam_sharpness",
        "relative_flatness",
    ]
    if args.dataset == "TinyImageNet" or "CIFAR100":
        sharpness_dataloader = data_loader_manager.get_sharpness_loader()
        if args.dataset == "TinyImageNet":
            mertics = [
                "acc",
                "ece",
                "squared_euclidean_norm",
                "fisher_rao_norm",
                "sam_sharpness",
            ]

    if os.path.exists(sharpness_path):
        with open(sharpness_path) as f:
            sharpness_dict = json.load(f)
        for metric in mertics:
            if metric not in sharpness_dict["model_eot"]:
                dummy_model = deepcopy(model)
                if (
                    (args.dataset == "TinyImageNet" and metric == "sam_sharpness")
                    or (
                        args.dataset == "TinyImageNet" and metric == "relative_flatness"
                    )
                    or (args.dataset == "CIFAR100" and metric == "relative_flatness")
                ):
                    Metrics_Processor = sm.MetricsProcessor(
                        [metric],
                        dummy_model,
                        sharpness_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )
                else:
                    if args.dataset == "NAMES":
                        Metrics_Processor = lsm.MetricsProcessor(
                            [metric],
                            dummy_model,
                            train_dataloader,
                            test_dataloader,
                            device,
                            args.seed,
                            args.model_name,
                            data_loader_manager.num_classes,
                        )
                    else:
                        Metrics_Processor = sm.MetricsProcessor(
                            [metric],
                            dummy_model,
                            train_dataloader,
                            test_dataloader,
                            device,
                            args.seed,
                            args.model_name,
                            data_loader_manager.num_classes,
                        )
                sharpness_dict["model_eot"][f"{metric}"] = (
                    Metrics_Processor.compute_metrics()[f"{metric}"]
                )
                with open(sharpness_path, "w") as f:
                    json.dump(sharpness_dict, f)
        print(sharpness_dict)
    else:
        sharpness_dict = {}
        sharpness_dict["model_eot"] = {}
        for metric in mertics:
            dummy_model = deepcopy(model)
            if (
                (args.dataset == "TinyImageNet" and metric == "sam_sharpness")
                or (args.dataset == "TinyImageNet" and metric == "relative_flatness")
                or (args.dataset == "CIFAR100" and metric == "relative_flatness")
            ):
                Metrics_Processor = sm.MetricsProcessor(
                    [metric],
                    dummy_model,
                    sharpness_dataloader,
                    test_dataloader,
                    device,
                    args.seed,
                    args.model_name,
                    data_loader_manager.num_classes,
                )
            else:
                if args.dataset == "NAMES":
                    Metrics_Processor = lsm.MetricsProcessor(
                        [metric],
                        dummy_model,
                        train_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )
                else:
                    Metrics_Processor = sm.MetricsProcessor(
                        [metric],
                        dummy_model,
                        train_dataloader,
                        test_dataloader,
                        device,
                        args.seed,
                        args.model_name,
                        data_loader_manager.num_classes,
                    )

            sharpness_dict["model_eot"][f"{metric}"] = (
                Metrics_Processor.compute_metrics()[f"{metric}"]
            )
            with open(sharpness_path, "w") as f:
                json.dump(sharpness_dict, f)


def corruption_test(model_path, args, device):
    print("Corruption Experiment")
    print("Loading model")
    if "CIFAR10-C" in args.dataset or "CIFAR100-C" in args.dataset:
        dummy_loader_manager = CIFAR_dataloader.DataLoaderManagerCIFAR(
            config=args,
            dataset_name=args.dataset,
            seed=args.seed,
        )
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

    elif "TinyImageNet-C" in args.dataset:
        dummy_loader_manager = IMAGENET_dataloader.DataLoaderManagerImageNet(
            config=args,
            dataset_name=args.dataset,
            seed=args.seed,
        )

        corruptions = [
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

    model, criterion, optimizer, scheduler = instantiate_model(
        args, dummy_loader_manager, device
    )
    model.load_state_dict(torch.load(model_path))
    corruptions_dict = {}
    for corruption in corruptions:
        corruptions_dict[corruption] = {}
        for severity in range(1, 6):
            if "CIFAR10-C" in args.dataset or "CIFAR100-C" in args.dataset:
                data_loader_manager = CIFAR_dataloader.DataLoaderManagerCIFAR(
                    config=args,
                    dataset_name=args.dataset,
                    seed=args.seed,
                    corruption=corruption,
                    corruption_severity=severity,
                )
            if "TinyImageNet-C" in args.dataset:
                data_loader_manager = IMAGENET_dataloader.DataLoaderManagerImageNet(
                    config=args,
                    dataset_name=args.dataset,
                    seed=args.seed,
                    corruption=corruption,
                    corruption_severity=severity,
                )
            corrupt_dataloader = data_loader_manager.get_corrupted_dataset()
            corruptions_dict[corruption][f"{severity}"] = mm.evaluate(
                model,
                corrupt_dataloader,
                criterion,
                data_loader_manager.num_classes,
                device,
            )
    corrupt_pth = f"{args.models_dir}/{args.save_name}_corruption.json"
    with open(f"{corrupt_pth}", "w") as f:
        json.dump(corruptions_dict, f)


def main(args):
    print("-----------------")
    print(f"Experirmental Setup:")
    print("-----------------")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-----------------")

    set_seed(args.seed)
    device = get_device()
    sharpness_metrics_complete = True

    if args.corrupt == True:
        dataset_dir = args.dataset.replace("-C", "")
        args.models_dir = Path(args.models_dir) / dataset_dir
        args.models_dir = args.models_dir / args.save_name / str(args.seed)
        model_path = f"{args.models_dir}/{args.save_name}.pth"
        corruption_test(model_path, args, device)

    else:
        args.models_dir = Path(args.models_dir) / args.dataset
        args.models_dir = args.models_dir / args.save_name / str(args.seed)
        if args.innit == False:
            model_path = f"{args.models_dir}/{args.save_name}.pth"
        elif args.innit == True:
            model_path = f"{args.models_dir}/initialisation.pth"
            args.save_name = "innit"

        if (
            args.dataset == "CIFAR10"
            or args.dataset == "CIFAR10R"
            or args.dataset == "CIFAR100"
            or args.dataset == "CIFAR100R"
        ):
            data_loader_manager = CIFAR_dataloader.DataLoaderManagerCIFAR(
                config=args,
                dataset_name=args.dataset,
                seed=args.seed,
                rand_prob=args.rand_prob,
            )
        elif args.dataset == "NAMES":
            data_loader_manager = NAMES_dataloader.DataLoaderManagerNAMES(
                config=args, dataset_name=args.dataset, seed=args.seed, device=device
            )
        elif "TinyImageNet" in args.dataset:
            data_loader_manager = IMAGENET_dataloader.DataLoaderManagerImageNet(
                config=args,
                dataset_name=args.dataset,
                seed=args.seed,
            )

        if args.sharpness:
            sharpness_path = f"{args.models_dir}/{args.save_name}_sharpness.json"
            mertics = [
                "acc",
                "ece",
                "squared_euclidean_norm",
                "fisher_rao_norm",
                "sam_sharpness",
                "relative_flatness",
            ]

        if args.dataset == "TinyImageNet":
            mertics = [
                "acc",
                "ece",
                "squared_euclidean_norm",
                "fisher_rao_norm",
                "sam_sharpness",
            ]

        if os.path.isfile(sharpness_path):
            with open(sharpness_path) as f:
                sharpness_dict = json.load(f)
            for metric in mertics:
                if metric not in sharpness_dict["model_eot"]:
                    sharpness_metrics_complete = False
        else:
            sharpness_metrics_complete = False

        if os.path.isfile(model_path) == True and sharpness_metrics_complete == False:
            calculate_sharpness(
                args, data_loader_manager, model_path, sharpness_path, device
            )

        else:
            train_model(args, data_loader_manager, device)

    print("Finished")


if __name__ == "__main__":
    args = options_parser()

    main(args)
