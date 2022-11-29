import time
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.moon.ClientTrainer import ClientTrainer
from algorithms.moon.criterion import ModelContrastiveLoss
from algorithms.BaseServer import BaseServer
from algorithms.measures import *

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
        moon_criterion = ModelContrastiveLoss(algo_params.mu, algo_params.tau)

        self.client = ClientTrainer(
            moon_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        self.prev_locals = []
        self._init_prev_locals()

        print("\n>>> MOON Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        for round_idx in range(self.n_rounds):

            # Initial Model Statistics
            if round_idx == 0:
                test_acc = evaluate_model(
                    self.model, self.testloader, device=self.device
                )
                self.server_results["test_accuracy"].append(test_acc)

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )

            # Get aggregated weights & update global
            ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)

    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.prev_locals[client_idx]
            )

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            for local_weights, client in zip(updated_local_weights, sampled_clients):
                self.prev_locals[client] = local_weights

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _init_prev_locals(self):
        weights = self.model.state_dict()
        for _ in range(self.n_clients):
            self.prev_locals.append(copy.deepcopy(weights))
