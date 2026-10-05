import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def sphere_function(x, y):
    return x**2 + y**2


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rastrigin_function(x, y, A=10):
    return A * 2 + (
        x**2 - A * np.cos(2 * np.pi * x) + (y**2 - A * np.cos(2 * np.pi * y))
    )


def beale_function(x, y):
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def booth_function(x, y):
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def three_hump_camel_function(x, y):
    return 2 * x**2 - 1.05 * x**4 + (x**6 / 6) + x * y + y**2


def himmelblaus_function(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


@torch.no_grad()
def sam_sharpness(
    model,
    train_inputs,
    train_targets,
    samples=1000,
    ro=0.005,
    seed: int = 0,
    device=torch.device("cpu"),
):
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    model.eval()
    num_params = sum(torch.numel(x) for x in model.parameters())

    def get_loss(model):
        loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            output = model(train_inputs)
        return loss_fn(output, train_targets).item()

    original_loss = get_loss(model)
    new_losses = []
    for _ in range(samples):
        new_state_dict = deepcopy(model.state_dict())
        for param_name in new_state_dict:
            if new_state_dict[param_name].dtype != torch.int64:
                new_state_dict[param_name] += (
                    torch.randn(
                        new_state_dict[param_name].size(),
                        dtype=new_state_dict[param_name].dtype,
                        layout=new_state_dict[param_name].layout,
                        device=new_state_dict[param_name].device,
                        generator=rng,
                    )
                    / (num_params**0.5)
                    * ro
                )
        new_model = deepcopy(model)
        new_model.load_state_dict(new_state_dict)

        loss = get_loss(new_model)

        new_losses.append((loss - original_loss) / ro)
    return {
        "mean": np.mean(np.abs(new_losses)),
        "std": np.std(np.abs(new_losses)),
        "median": np.median(np.abs(new_losses)),
        "max": np.max(np.abs(new_losses)),
        "min": np.min(np.abs(new_losses)),
    }


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, return_feat=False):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        if return_feat:
            return self.fc3(x), x
        return self.fc3(x)


def eval_model(model, inputs, targets):
    model.eval()
    criterion = nn.MSELoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    return loss.item()


# Training the model
def train_model(
    model,
    train_inputs,
    train_targets,
    test_inputs,
    test_targets,
    epochs=100_000,
    target_epochs=[100_000, 10_000, 1_000, 1_00, 10, 1],
    lr=1e-3,
    path=".",
    name="test",
    seed=0,
    device=torch.device("cpu"),
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch = 0
    losses = []
    model.eval()
    train_loss = eval_model(model, train_inputs, train_targets)
    test_loss = eval_model(model, test_inputs, test_targets)
    results = {}
    for rho in [0.0005, 0.005, 0.05, 0.5]:
        results[f"sam_{rho}"] = sam_sharpness(
            model, train_inputs, train_targets, ro=rho, device=device
        )
    results["train_loss"] = train_loss
    results["test_loss"] = test_loss
    results["gen_gap"] = test_loss - train_loss
    results["epoch"] = epoch
    torch.save(results, f"{path}/sharpness/{name}_seed_{seed}_epoch_{epoch}.pt")
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        if epoch in target_epochs:
            with torch.no_grad():
                torch.save(
                    model.state_dict(),
                    f"{path}/models/{name}_seed_{seed}_epoch_{epoch}.pt",
                )
                model.eval()
                train_loss = eval_model(model, train_inputs, train_targets)
                test_loss = eval_model(model, test_inputs, test_targets)
                results = {}
                for rho in [0.0005, 0.005, 0.05, 0.5]:
                    results[f"sam_{rho}"] = sam_sharpness(
                        model, train_inputs, train_targets, ro=rho, device=device
                    )
                results["train_loss"] = train_loss
                results["test_loss"] = test_loss
                results["gen_gap"] = test_loss - train_loss
                results["epoch"] = epoch
                torch.save(
                    results, f"{path}/sharpness/{name}_seed_{seed}_epoch_{epoch}.pt"
                )
        losses.append(loss.item())
    torch.save(losses, f"{path}/stats/{name}_seed_{seed}.pt")
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--objective",
        choices=[
            "sphere",
            "booth",
            "three_hump_camel",
            "himmelblaus",
            "rosenbrock",
            "beale",
            "rastrigin",
        ],
        required=True,
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    single_objectives = {
        "sphere": sphere_function,
        "booth": booth_function,
        "three_hump_camel": three_hump_camel_function,
        "himmelblaus": himmelblaus_function,
        "rosenbrock": rosenbrock_function,
        "beale": beale_function,
        "rastrigin": rastrigin_function,
    }

    domain = 3.5
    n_samples = 10_000
    os.makedirs(f"{args.path}/data/", exist_ok=True)
    if os.path.exists(f"{args.path}/data/{args.seed}.pt"):
        data = torch.load(f"{args.path}/data/{args.seed}.pt", weights_only=False)
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
        train_inputs = data["train_inputs"]
        test_inputs = data["test_inputs"]
    else:
        x_train = np.random.uniform(-domain, domain, size=(n_samples, 1))
        y_train = np.random.uniform(-domain, domain, size=(n_samples, 1))
        train_inputs = np.hstack((x_train, y_train))

        x_test = np.random.uniform(-domain, domain, size=(n_samples, 1))
        y_test = np.random.uniform(-domain, domain, size=(n_samples, 1))
        test_inputs = np.hstack((x_test, y_test))
        # Remove duplicates from the test set
        del_idx = []
        for idx, test in enumerate(tqdm(test_inputs)):
            if test in train_inputs:
                del_idx.append(idx)
        test_inputs = np.delete(test_inputs, del_idx, axis=0)

        torch.save(
            {
                "train_inputs": train_inputs,
                "test_inputs": test_inputs,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            },
            f"{args.path}/data/{args.seed}.pt",
        )

    model = MLP()
    os.makedirs(f"{args.path}/models", exist_ok=True)
    os.makedirs(f"{args.path}/sharpness", exist_ok=True)
    os.makedirs(f"{args.path}/stats", exist_ok=True)
    if os.path.exists(f"{args.path}/models/model.pt"):
        model.load_state_dict(
            torch.load(f"{args.path}/models/model.pt", weights_only=True)
        )
    else:
        torch.save(model.state_dict(), f"{args.path}/models/model.pt")

    train_targets = single_objectives[args.objective](x_train, y_train)
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32, device=device)
    train_targets = torch.tensor(train_targets, dtype=torch.float32, device=device)

    test_targets = single_objectives[args.objective](x_test, y_test)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_targets = torch.tensor(test_targets, dtype=torch.float32, device=device)
    model = MLP()
    model.load_state_dict(torch.load(f"{args.path}/models/model.pt", weights_only=True))
    model.to(device)
    train_model(
        model,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        path=args.path,
        name=args.objective,
        seed=args.seed,
        device=device,
    )


if __name__ == "__main__":
    main()
