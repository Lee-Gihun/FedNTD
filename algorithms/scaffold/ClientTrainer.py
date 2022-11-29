import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.scaffold.utils import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        self.dg_model = None
        self.step_count = 0
        self.c, self.c_i = 0, 0
        self.adaptive_divison = self.algo_params.adaptive_divison

    def train(self):
        """Local training"""

        # Keep global model weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self._scaffold_step(data, targets)

        # update control variates for scaffold algorithm
        c_i_plus, c_update_amount = self._update_control_variate()

        local_results = self._get_local_stats()

        return local_results, local_size, c_i_plus, c_update_amount

    def download_global(self, server_weights, server_optimizer, c, c_i):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self.c, self.c_i = c.to(self.device), c_i.to(self.device)

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.step_count = 0

    def _scaffold_step(self, data, targets):
        self.optimizer.zero_grad()

        # forward pass
        data, targets = data.to(self.device), targets.to(self.device)
        logits = self.model(data)
        loss = self.criterion(logits, targets)

        # backward pass
        loss.backward()
        grad_batch = flatten_grads(self.model).detach().clone()
        self.optimizer.zero_grad()

        # add control variate
        grad_batch = grad_batch - self.c_i + self.c
        self.model = assign_grads(self.model, grad_batch)
        self.optimizer.step()
        self.step_count += 1

    @torch.no_grad()
    def _update_control_variate(self):

        divisor = self.__get_divisor()

        server_params = flatten_weights(self.dg_model)
        local_params = flatten_weights(self.model)
        param_move = server_params - local_params

        c_i_plus = self.c_i.cpu() - self.c.cpu() + (divisor * param_move)
        c_update_amount = c_i_plus - self.c_i.cpu()

        return c_i_plus, c_update_amount

    def __get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def __get_divisor(self):
        local_lr = self.__get_learning_rate()
        K = self.step_count
        for param_group in self.optimizer.param_groups:
            rho = param_group["momentum"]
        new_K = (K - rho * (1.0 - pow(rho, K)) / (1.0 - rho)) / (1.0 - rho)

        if self.adaptive_divison:
            divisor = 1.0 / (new_K * local_lr)
        else:
            divisor = 1.0 / (K * local_lr)

        return divisor
