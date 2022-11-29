import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.feddyn.utils import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        # FedDyn setting
        self.dyn_alpha = self.algo_params.dyn_alpha
        self.previous_grads = None
        self.server_weights = None

    def train(self):
        """Local training"""

        self.model.train()
        self.model.to(self.device)

        local_results = {}
        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, targets)

                # Feddyn penalty terms
                lin_penalty = 0.0
                sq_penalty = 0.0

                for name, param in self.model.named_parameters():
                    lin_penalty += torch.sum(
                        torch.mul(self.previous_grads[name], param)
                    )
                    sq_penalty += torch.sum(
                        torch.square(self.server_weights[name] - param)
                    )

                loss = loss - lin_penalty + 0.5 * self.dyn_alpha * sq_penalty

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer, previous_local_grads):
        """
        Load model & Optimizer
        For feddyn, also load state_dict itself.
        """
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

        self.previous_grads = previous_local_grads
        self.server_weights = {}
        for k, v in server_weights.items():
            self.server_weights[k] = v.clone().detach()

    def upload_local(self, server_weights):
        """
        Uploads local model's parameters
        local_weight and local_grads are saved as state_dict format, on gpu.
        """
        local_weights = detached_statedict(self.model.state_dict())

        local_grads = detached_statedict(self.previous_grads)
        for k in local_grads.keys():
            local_grads[k] -= self.dyn_alpha * (local_weights[k] - server_weights[k])

        return local_weights, local_grads

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)

        self.previous_grads = None
        self.server_weights = None
