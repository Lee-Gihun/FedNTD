import torch
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.fedcurv.ClientTrainer import ClientTrainer
from algorithms.BaseServer import BaseServer

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        self.client = ClientTrainer(
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )
        
        # Dictionaries for saving 'diag(Hess)'(=ut) and 'diag(Hess)*local_weight'(=vt)
        self.updated_local_uts = {}
        self.updated_local_vts = {}

        print("\n>>> FedCurv Server initialized...\n")


    def _clients_training(self, sampled_clients, round_idx):
        """
        Conduct local training and get trained local models' weights
        Now _clients_training function takes round_idx
        (Since we can not use Fisher regularization on the very first round; round_idx=0)
        """
        
        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Unless the round > 0, we don't have Fisher regularizer
        if round_idx != 0:
            # Get global Ut and Vt
            with torch.no_grad():
                Ut = torch.sum(
                    torch.stack(list(self.updated_local_uts.values())), dim=0
                )
                Vt = torch.sum(
                    torch.stack(list(self.updated_local_vts.values())), dim=0
                )

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Download Fisher regularizer
            if round_idx != 0:
                with torch.no_grad():
                    Pt = Ut
                    Qt = Vt

                    if client_idx in self.updated_local_vts:
                        Pt -= self.updated_local_uts[client_idx]
                        Qt -= self.updated_local_vts[client_idx]

                self.client.download_fisher_regularizer(Pt, Qt)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Upload 'diag(Hess)'(=ut) and 'diag(Hess) dot optimized weight'(=vt)
            # Uploaded vector is stored at the dictionary, having client_idx as the key
            local_ut, local_vt = self.client.upload_local_fisher()
            self.updated_local_uts[client_idx] = local_ut
            self.updated_local_vts[client_idx] = local_vt

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results
