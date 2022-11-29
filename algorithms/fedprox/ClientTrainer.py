import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """

        self.mu = self.algo_params.mu

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

                # Add proximal loss term
                loss += self._proximal_term(self.dg_model, self.model, self.mu)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()

        return local_results, local_size

    def _proximal_term(self, dg_model, model, mu):
        """Proximal regularizer of FedProx"""

        vec = []
        for _, ((name1, param1), (name2, param2)) in enumerate(
            zip(model.named_parameters(), dg_model.named_parameters())
        ):
            if name1 != name2:
                raise RuntimeError
            else:
                vec.append((param1 - param2).view(-1, 1))

        all_vec = torch.cat(vec)
        square_term = torch.square(all_vec).sum()
        proximal_loss = 0.5 * mu * square_term

        return proximal_loss
