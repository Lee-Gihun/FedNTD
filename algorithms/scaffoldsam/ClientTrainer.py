import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.optimizer import SAM
from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.scaffoldsam.utils import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        self.sam_optimizer = None
        self.rho = self.algo_params.rho
        self.adaptive = self.algo_params.adaptive
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
        self.sam_optimizer = self._get_sam_optimizer(self.optimizer)
        self.c, self.c_i = c.to(self.device), c_i.to(self.device)

    def _get_sam_optimizer(self, base_optimizer):
        optim_params = base_optimizer.state_dict()
        lr = optim_params["param_groups"][0]["lr"]
        momentum = optim_params["param_groups"][0]["momentum"]
        weight_decay = optim_params["param_groups"][0]["weight_decay"]
        sam_optimizer = SAM(
            self.model.parameters(),
            base_optimizer=torch.optim.SGD,
            rho=self.rho,
            adaptive=self.adaptive,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        return sam_optimizer

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.step_count = 0

    def _scaffold_step(self, data, targets):
        # forward pass
        data, targets = data.to(self.device), targets.to(self.device)

        enable_running_stats(self.model)
        logits = self.model(data)
        loss = self.criterion(logits, targets)
        loss.backward()
        grad_batch = flatten_grads(self.model).detach().clone()
        disable_running_stats(self.model)
        grad_batch = None

        # scaffold grad batch
        grad_batch = -self.c_i + self.c
        self.model = assign_grads(self.model, grad_batch)

        # first ascent step
        self.sam_optimizer.first_step(zero_grad=True)

        # second descent step
        self.criterion(self.model(data), targets).backward()
        self.sam_optimizer.second_step(zero_grad=True)

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
