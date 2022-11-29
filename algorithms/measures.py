import torch
import torch.nn.functional as F
from .utils import *

__all__ = ["evaluate_model", "evaluate_model_classwise", "get_round_personalized_acc"]


@torch.no_grad()
def evaluate_model(model, dataloader, device="cuda:0"):
    """Evaluate model accuracy for the given dataloader"""
    model.eval()
    model.to(device)

    running_count = 0
    running_correct = 0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        pred = logits.max(dim=1)[1]

        running_correct += (targets == pred).sum().item()
        running_count += data.size(0)

    accuracy = round(running_correct / running_count, 4)

    return accuracy


@torch.no_grad()
def evaluate_model_classwise(
    model, dataloader, num_classes=10, device="cuda:0",
):
    """Evaluate class-wise accuracy for the given dataloader."""

    model.eval()
    model.to(device)

    classwise_count = torch.Tensor([0 for _ in range(num_classes)]).to(device)
    classwise_correct = torch.Tensor([0 for _ in range(num_classes)]).to(device)

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        preds = logits.max(dim=1)[1]

        for class_idx in range(num_classes):
            class_elem = targets == class_idx
            classwise_count[class_idx] += class_elem.sum().item()
            classwise_correct[class_idx] += (targets == preds)[class_elem].sum().item()

    classwise_accuracy = classwise_correct / classwise_count
    accuracy = round(classwise_accuracy.mean().item(), 4)

    return classwise_accuracy.cpu(), accuracy


@torch.no_grad()
def get_round_personalized_acc(round_results, server_results, data_distributed):
    """Evaluate personalized FL performance on the sampled clients."""

    sampled_clients = server_results["client_history"][-1]
    local_dist_list, local_size_list = sampled_clients_identifier(
        data_distributed, sampled_clients
    )

    local_cwa_list = round_results["classwise_accuracy"]

    result_dict = {}

    in_dist_acc_list, out_dist_acc_list = [], []
    local_size_prop = F.normalize(torch.Tensor(local_size_list), dim=0, p=1)

    for local_cwa, local_dist_vec in zip(local_cwa_list, local_dist_list):
        local_dist_vec = torch.Tensor(local_dist_vec)
        inverse_dist_vec = calculate_inverse_dist(local_dist_vec)
        in_dist_acc = torch.dot(local_cwa, local_dist_vec)
        in_dist_acc_list.append(in_dist_acc)
        out_dist_acc = torch.dot(local_cwa, inverse_dist_vec)
        out_dist_acc_list.append(out_dist_acc)

    round_in_dist_acc = torch.Tensor(in_dist_acc_list)
    round_out_dist_acc = torch.Tensor(out_dist_acc_list)

    result_dict["in_dist_acc_prop"] = torch.dot(
        round_in_dist_acc, local_size_prop
    ).item()
    result_dict["in_dist_acc_mean"] = (round_in_dist_acc).mean().item()
    result_dict["in_dist_acc_std"] = (round_in_dist_acc).std().item()
    result_dict["out_dist_acc"] = torch.dot(round_out_dist_acc, local_size_prop).item()
    result_dict["in_dout_acc_mean"] = (round_out_dist_acc).mean().item()

    return result_dict


@torch.no_grad()
def calculate_inverse_dist(dist_vec):
    """Get the out-local distribution"""
    inverse_dist_vec = (1 - dist_vec) / (dist_vec.nelement() - 1)

    return inverse_dist_vec
