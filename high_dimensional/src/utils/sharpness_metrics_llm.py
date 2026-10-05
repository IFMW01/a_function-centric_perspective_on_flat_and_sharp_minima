import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import math
from copy import deepcopy
import numpy as np
import torch.nn as nn
from numpy import linalg as LA
from pyhessian import hessian
from torch.autograd import Variable, grad
from tqdm import tqdm
from torchmetrics.classification import MulticlassCalibrationError


class MetricsProcessor:
    def __init__(
        self,
        metrics,
        model,
        train_dataloader,
        test_dataloader,
        device,
        seed,
        model_name,
        num_classes,
    ) -> None:
        self.metrics = metrics
        self.model = model.eval()
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.seed = seed
        self.model_name = model_name

        self.cache_max_hessian_eigenvalue = None
        self.cache_hessian_trace = None
        self.num_classes = num_classes

    def compute_metrics(self):
        """
        Compute metrics
        """

        results_dict = {}

        self.model.eval()
        for metric in self.metrics:
            compute_func = getattr(self, metric)
            print(f"Running metrics {str(metric)}...")
            results_dict[metric] = compute_func()
            self.model.zero_grad(set_to_none=True)

        return results_dict

    # --------------------------------------------------------------------------
    def run_model(self):
        self.model.eval()
        avg_loss = torch.nn.CrossEntropyLoss()

        all_outputs = []
        all_y = []
        all_feat = []
        with torch.no_grad():
            for batch in self.train_dataloader:
                y = batch["labels"]

                output = self.model(**batch)
                feat = output.hidden_states[-1]

                all_outputs.append(output.logits)
                all_feat.append(feat)
                all_y.append(y)

        all_outputs = torch.cat(all_outputs)

        all_feat = torch.cat(all_feat)
        all_y = torch.cat(all_y)
        train_loss = avg_loss(all_outputs, all_y)

        return train_loss, all_outputs, all_feat, all_y

    @torch.no_grad()
    def sam_sharpness(self):
        self.model.eval()
        num_params = sum(torch.numel(x) for x in self.model.parameters())

        def get_loss(model):
            losses = []
            with torch.no_grad():
                for batch in self.train_dataloader:
                    y = batch["labels"]

                    output = self.model(**batch)
                    losses.append(output.loss.item())

            return np.mean(losses)

        original_loss = get_loss(self.model)
        new_losses = []
        for _ in tqdm(range(100)):
            new_state_dict = deepcopy(self.model.state_dict())

            ro = 0.005

            for param_name in new_state_dict:
                if new_state_dict[param_name].dtype != torch.int64:
                    new_state_dict[param_name] += (
                        torch.randn_like(new_state_dict[param_name])
                        / (num_params**0.5)
                        * ro
                    )

            new_model = deepcopy(self.model)
            new_model.load_state_dict(new_state_dict)

            loss = get_loss(new_model)

            new_losses.append((loss - original_loss) / ro)
        return np.mean(np.abs(new_losses))

    def hessian_single_layer(self, layer, train_loss):
        # hessian calculation for the layer of interest
        last_layer_jacobian = grad(
            train_loss, layer, create_graph=True, retain_graph=True
        )
        hessian = []
        for n_grd in last_layer_jacobian[0]:
            for w_grd in n_grd:
                drv2 = grad(w_grd, layer, retain_graph=True)
                hessian.append(drv2[0].data.cpu().numpy().flatten())

        return hessian

    def get_feature_layer(self):
        params = list(self.model.parameters())
        feature_layer_idx = -1
        for i in range(len(params)):
            if params[i] is self.model.score:
                feature_layer_idx = i

        assert i is not -1
        feature_layer = list(self.model.parameters())[feature_layer_idx]
        return feature_layer, feature_layer_idx

    def squared_euclidean_norm(self):
        layer, _ = self.get_feature_layer()

        weights_norm = 0.0
        for n in layer.data.cpu().numpy():
            for w in n:
                weights_norm += w**2
        # print("Squared euclidian norm is calculated", weights_norm)
        return float(weights_norm)

    def max_hessian_eigenvalue_slow(self, hessian=None):
        if hessian is None:
            layer, _ = self.get_feature_layer()
            train_loss, _, _, _ = self.run_model()
            hessian = self.hessian_single_layer(layer, train_loss)

        max_eignv = LA.eigvalsh(hessian)[-1]
        return max_eignv

    def hessian_trace_slow(self, hessian=None):
        if hessian is None:
            layer, _ = self.get_feature_layer()
            train_loss, _, _, _ = self.run_model()
            hessian = self.hessian_single_layer(layer, train_loss)

        trace = np.trace(hessian)
        return trace

    def calc_hessian_metrics(self):
        if self.cache_max_hessian_eigenvalue is not None:
            return

        hessian_comp = hessian(
            self.model,
            torch.nn.CrossEntropyLoss(),
            dataloader=self.train_dataloader,
            cuda=True,
        )

        top_eigenvalues, _ = hessian_comp.eigenvalues(maxIter=20, tol=0.01)
        trace = hessian_comp.trace(maxIter=20, tol=0.01)

        self.cache_max_hessian_eigenvalue = top_eigenvalues[0]
        self.cache_hessian_trace = np.mean(trace)

    def max_hessian_eigenvalue(self):
        self.calc_hessian_metrics()
        return self.cache_max_hessian_eigenvalue

    def hessian_trace(self):
        self.calc_hessian_metrics()
        return self.cache_hessian_trace

    def count_layers(self, model):
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

    def fisher_rao_norm(self):
        model = self.model.eval()
        with torch.no_grad():
            train_loss, train_output, activation, labels = self.run_model()
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
            ((self.count_layers(self.model) + 1) ** 2)
            * (1.0 / len(train_output))
            * sum_derivatives
        )
        return fr_norm

    def calculateNeuronwiseHessians_fc_layer(
        self, feature_layer, train_loss, alpha, normalize=False
    ):
        shape = feature_layer.shape

        layer_jacobian = grad(
            train_loss, feature_layer, create_graph=True, retain_graph=True
        )
        layer_jacobian_out = layer_jacobian[0]
        drv2 = Variable(
            torch.empty(shape[1], shape[0], shape[0], shape[1]), requires_grad=True
        ).to(self.device)
        for ind, n_grd in enumerate(layer_jacobian[0].T):
            for neuron_j in range(shape[0]):
                drv2[ind][neuron_j] = grad(
                    n_grd[neuron_j].to(self.device), feature_layer, retain_graph=True
                )[0].to(self.device)
        # print("got hessian")

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

    def relative_flatness(self):
        with torch.no_grad():
            train_loss, train_output, activation, labels = self.run_model()

        feature_layer, feature_layer_idx = self.get_feature_layer()

        activation = activation.detach().cpu().numpy()

        activation = np.squeeze(activation)
        sigma = np.std(activation, axis=0)

        j = 0
        for p in self.model.parameters():
            if feature_layer_idx - 2 == j or feature_layer_idx - 1 == j:
                for i, sigma_i in enumerate(sigma):
                    if sigma_i != 0.0:
                        p.data[i] = p.data[i] / sigma_i
            if feature_layer_idx == j:
                for i, sigma_i in enumerate(sigma):
                    p.data[:, i] = p.data[:, i] * sigma_i
                feature_layer = p
            j += 1

        # train_loss, _, _, _ = self.run_model()

        trace_nm = 0

        for batch in self.train_dataloader:
            label = batch["labels"]
            output = self.model(**batch)
            softmax = torch.softmax(output.logits, dim=1)
            train_loss = output.loss

            curr_trace_nm, curr_maxeigen_nm = self.calculateNeuronwiseHessians_fc_layer(
                feature_layer, train_loss, None, normalize=False
            )

            trace_nm += curr_trace_nm

            self.model.zero_grad(set_to_none=True)

        return float(trace_nm)

        # print("Neuronwise tracial measure is", trace_nm)
        # print("Neuronwise max eigenvalue measure is", maxeigen_nm)

    @torch.no_grad()
    def ece(self):
        self.model.eval()
        ece = MulticlassCalibrationError(self.num_classes, n_bins=15, norm="l1")
        for batch in self.test_dataloader:
            label = batch["labels"]
            output = self.model(**batch)
            softmax = torch.softmax(output.logits, dim=1)
            ece.update(softmax, label)
        ece = ece.compute().item()

        return ece

    @torch.no_grad()
    def acc(self):
        self.model.eval()
        correct = 0
        total = 0

        for batch in self.test_dataloader:
            label = batch["labels"]
            output = self.model(**batch)
            softmax = torch.softmax(output.logits, dim=1)
            comparison_with_gold = torch.argmax(softmax, dim=-1) == label
            correct += comparison_with_gold.sum().item()
            total += len(label)

        acc = correct / total
        return acc
