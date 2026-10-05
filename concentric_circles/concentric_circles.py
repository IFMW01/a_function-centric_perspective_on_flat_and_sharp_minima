import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import math
import os
import random

import numpy as np
import sklearn.datasets as ds
from numpy import linalg as LA

import torch
import torch.nn as nn
from torch.autograd import Variable, grad

torch.set_num_threads(1)  # For intra-op parallelism
torch.set_num_interop_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(2, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.output = torch.nn.Linear(100, 2)

    def forward(self, x, return_feat=False):
        x = torch.nn.functional.relu(self.fc1(x))
        features = torch.nn.functional.relu(self.fc2(x))
        x = self.output(features)
        if return_feat:
            return x, features
        return x


class GuassianData(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels[idx]


def train(net, dataloader, eval_dataloader, epochs, criterion, optimizer, device):
    for epoch in range(epochs):  # loop over the dataset multiple times
        for inputs, labels in dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate(test_data, model, device):
    running_loss = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for iters, data in enumerate(test_data):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.flatten()).sum().item()
    return (
        100 * correct / total,
        running_loss / (iters + 1),
    )


def run_model(model, train_dataloader, device):
    model = model.eval()
    avg_loss = torch.nn.CrossEntropyLoss()

    all_outputs = []
    all_y = []
    all_feat = []
    with torch.no_grad():
        for inputs, labels in train_dataloader:
            X = inputs.to(device)
            y = labels.to(device)

            output, feat = model(X, return_feat=True)

            all_outputs.append(output)
            all_feat.append(feat)
            all_y.append(y)

    all_outputs = torch.cat(all_outputs)
    all_feat = torch.cat(all_feat)
    all_y = torch.cat(all_y)

    train_loss = avg_loss(all_outputs, all_y)

    return train_loss, all_outputs, all_feat, all_y


def count_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.Linear,
                torch.nn.Conv1d,
                torch.nn.Conv2d,
                torch.nn.Conv3d,
                torch.nn.Embedding,
            ),
        ):
            count += 1
    return count


def fisher_rao_norm(model, train_dataloader, device):
    model = model.eval()
    with torch.no_grad():
        train_loss, train_output, activation, labels = run_model(
            model, train_dataloader, device
        )
    # calculate FisherRao norm
    # analytical formula for crossentropy loss from Appendix of the original paper
    sum_derivatives = 0
    m = torch.nn.Softmax(dim=0)
    for inp in range(len(train_output)):
        sum_derivatives += (
            np.inner(
                m(train_output[inp]).data.cpu().numpy(),
                train_output[inp].data.cpu().numpy(),
            )
            - train_output[inp].data.cpu().numpy()[labels[inp]]
        ) ** 2
    fr_norm = math.sqrt(
        ((count_layers(model) + 1) ** 2) * (1.0 / len(train_output)) * sum_derivatives
    )
    return fr_norm


def get_feature_layer(model):
    params = list(model.parameters())
    feature_layer_idx = -1
    for i in range(len(params)):
        if params[i] is model.output.weight:
            feature_layer_idx = i

    assert i != -1
    feature_layer = list(model.parameters())[feature_layer_idx]
    return feature_layer, feature_layer_idx


def calculateNeuronwiseHessians_fc_layer(
    feature_layer, train_loss, alpha, normalize=False, device=torch.device("cpu")
):
    shape = feature_layer.shape

    layer_jacobian = grad(
        train_loss, feature_layer, create_graph=True, retain_graph=True
    )
    layer_jacobian_out = layer_jacobian[0]
    drv2 = Variable(
        torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=False
    ).to(device)
    for ind, n_grd in enumerate(layer_jacobian[0].T):
        for neuron_j in range(shape[0]):
            drv2[ind][neuron_j] = grad(
                n_grd[neuron_j].to(device), feature_layer, retain_graph=True
            )[0].to(device)

    trace_neuron_measure = 0.0
    maxeigen_neuron_measure = 0.0
    for neuron_i in range(shape[0]):
        neuron_i_weights = feature_layer[neuron_i, :].data.cpu().numpy()
        for neuron_j in range(shape[0]):
            neuron_j_weights = feature_layer[neuron_j, :].data.cpu().numpy()
            hessian = drv2[:, neuron_j, neuron_i, :]
            trace = np.trace(hessian.data.cpu().numpy())
            if normalize:
                trace /= 1.0 * hessian.shape[0]
            trace_neuron_measure += neuron_i_weights.dot(neuron_j_weights) * trace
            if neuron_j == neuron_i:
                eigenvalues = LA.eigvalsh(hessian.data.cpu().numpy())
                maxeigen_neuron_measure += (
                    neuron_i_weights.dot(neuron_j_weights) * eigenvalues[-1]
                )
                # adding regularization term
                if alpha:
                    trace_neuron_measure += (
                        neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    )
                    maxeigen_neuron_measure += (
                        neuron_i_weights.dot(neuron_j_weights) * 2.0 * alpha
                    )

    return trace_neuron_measure, maxeigen_neuron_measure


def relative_flatness(model, train_dataloader, device):
    model = model.eval()
    with torch.no_grad():
        train_loss, train_output, activation, labels = run_model(
            model, train_dataloader, device
        )

    feature_layer, feature_layer_idx = get_feature_layer(model)

    activation = activation.detach().cpu().numpy()

    activation = np.squeeze(activation)
    sigma = np.std(activation, axis=0)

    j = 0
    for p in model.parameters():
        if feature_layer_idx - 2 == j or feature_layer_idx - 1 == j:
            for i, sigma_i in enumerate(sigma):
                if sigma_i != 0.0:
                    p.data[i] = p.data[i] / sigma_i
        if feature_layer_idx == j:
            for i, sigma_i in enumerate(sigma):
                p.data[:, i] = p.data[:, i] * sigma_i
            feature_layer = p
        j += 1

    trace_nm = 0

    avg_loss = torch.nn.CrossEntropyLoss()
    for inputs, labels in train_dataloader:
        X = inputs.to(device)
        y = labels.to(device)

        train_loss = avg_loss(model(X), y)

        curr_trace_nm, curr_maxeigen_nm = calculateNeuronwiseHessians_fc_layer(
            feature_layer, train_loss, None, normalize=False, device=device
        )

        trace_nm += curr_trace_nm

        model.zero_grad(set_to_none=True)

    return float(trace_nm)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    GENERATOR = torch.Generator()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--factor", type=float, required=True)
    results = []
    for seed in range(100):
        DEVICE = torch.device("cpu")
        batch_size = 5
        args = parser.parse_args()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        n = 100
        data, labels = ds.make_circles(
            n_samples=n, shuffle=True, noise=0.00, random_state=0, factor=args.factor
        )

        train_data = data[: n // 2]
        train_labels = labels[: n // 2]
        test_data = data[n // 2 :]
        test_labels = labels[n // 2 :]
        train_data = GuassianData(
            torch.tensor(train_data, dtype=torch.float, device=DEVICE),
            torch.tensor(train_labels, device=DEVICE),
        )
        test_data = GuassianData(
            torch.tensor(test_data, dtype=torch.float, device=DEVICE),
            torch.tensor(test_labels, device=DEVICE),
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=GENERATOR.manual_seed(0),
            num_workers=0,
        )
        eval_train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=GENERATOR.manual_seed(0),
            num_workers=0,
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=GENERATOR.manual_seed(0),
            num_workers=0,
        )

        model = MLP()
        os.makedirs("./models/init", exist_ok=True)
        os.makedirs("./models/perf", exist_ok=True)
        os.makedirs("./models/trained", exist_ok=True)

        model.load_state_dict(
            torch.load(f"./models/init/standard_init_seed_{seed}.pth")
        )

        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train(
            model,
            train_data_loader,
            eval_train_data_loader,
            300,
            criterion,
            optimizer,
            DEVICE,
        )
        train_acc, train_loss = evaluate(train_data_loader, model, DEVICE)
        test_acc, test_loss = evaluate(test_data_loader, model, DEVICE)
        fro = fisher_rao_norm(model, train_data_loader, DEVICE)
        relative_flatness_values = relative_flatness(model, train_data_loader, DEVICE)
        perfs = [
            train_acc,
            train_loss,
            test_acc,
            test_loss,
            fro,
            relative_flatness_values,
        ]
        results.append(perfs)

        if seed == 0:
            model = model.cpu()
            torch.save(
                model.state_dict(),
                f"./models/trained/factor_{args.factor}_seed_{seed}.pth",
            )

    torch.save(results, f"./models/perf/factor_{args.factor}.pth")


if __name__ == "__main__":
    main()
