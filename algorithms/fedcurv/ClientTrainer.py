import torch
from torch import autograd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer
from algorithms.fedcurv.utils import *

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        # Fisher regularization setting
        self.fisher_sample_size = self.algo_params.size
        self.fisher_lambda = self.algo_params.lam

        # Parameters for Fisher regularization
        self.enable_fisher_reg = False
        self.Pt, self.Qt = None, None

    def train(self):
        """Local training"""

        # Keep global model's weights
        self._keep_global()

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

                # Add Fisher regularizer to our criterion loss, if it is enabled
                if self.enable_fisher_reg:
                    all_params = flatten_weights(self.model, numpy_output=False)
                    reg_loss = self.fisher_lambda * torch.inner(
                        self.Pt, torch.square(all_params)
                    ) - self.fisher_lambda * 2.0 * torch.inner(self.Qt, all_params)
                    loss += reg_loss

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def download_fisher_regularizer(self, Pt, Qt):
        """Download regularizer coefficient from server"""
        self.enable_fisher_reg = True
        self.Pt = Pt
        self.Qt = Qt

    def upload_local_fisher(self):
        """
        Calculate ut and vt, and upload clone().detach() version of them
        Do Sampling to get fisher matrix on train data
        """
        local_params = flatten_weights(self.model, numpy_output=False).clone().detach()

        fisher_list = []
        samples_so_far = 0
        for data, targets in self.trainloader:
            batch_size = len(targets)

            data, targets = data.to(self.device), targets.to(self.device)
            output = self.model(data)
            crit = self.criterion(output, targets)
            samples_so_far += batch_size

            # Calculate flattend Gradient and save in the list
            grad = autograd.grad(crit, self.model.parameters())
            all_grad_eles = []
            for elewise in grad:
                all_grad_eles.append(elewise.view(-1))
            all_grad_eles = torch.cat(all_grad_eles)
            fisher_list.append(torch.square(all_grad_eles.clone().detach()))

            if samples_so_far > self.fisher_sample_size:
                break

        # Average (minibatch-wise) squared-gradient
        ut = torch.mean(torch.stack(fisher_list), dim=0).clone().detach()
        vt = torch.mul(ut, local_params).clone().detach()

        return ut, vt

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)

        # Set boolen token for enabling Fisher regularizer as false
        # Set Pt and Qt as None
        self.enable_fisher_reg = False
        self.Pt = None
        self.Qt = None
