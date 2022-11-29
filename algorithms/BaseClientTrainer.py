import torch
import torch.nn as nn
import copy

from .measures import *

__all__ = ["BaseClientTrainer"]


class BaseClientTrainer:
    def __init__(self, algo_params, model, local_epochs, device, num_classes):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        # algorithm-specific parameters
        self.algo_params = algo_params

        # model & optimizer
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.criterion = nn.CrossEntropyLoss()
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None
        self.num_classes = num_classes
        self.trainloader = None
        self.testloader = None

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, targets)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def finetune(self, server_weights, server_optimizer):
        tuning_results = {}

        # Download Global model and Optimizer
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)

        tuning_criterion = nn.CrossEntropyLoss()
        tuning_optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        tuning_optimizer.load_state_dict(server_optimizer)

        initial_acc = evaluate_model(self.model, self.testloader, self.device)
        tuning_results["finetune_0"] = initial_acc

        # Finetune downloaded models
        for tuning_epoch in range(1, 6):
            self.model.train()
            for data, targets in self.trainloader:
                tuning_optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = tuning_criterion(output, targets)

                # backward pass
                loss.backward()
                tuning_optimizer.step()

            tuning_acc = evaluate_model(self.model, self.testloader, self.device)
            tuning_results["finetune_%d" % tuning_epoch] = tuning_acc

        return tuning_results

    def _get_local_stats(self):
        local_results = {}

        local_results["train_acc"] = evaluate_model(
            self.model, self.trainloader, self.device
        )
        (
            local_results["classwise_accuracy"],
            local_results["test_acc"],
        ) = evaluate_model_classwise(
            self.model, self.testloader, self.num_classes, device=self.device,
        )

        return local_results

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)

    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.dg_model = copy.deepcopy(self.model)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False
