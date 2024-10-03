import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(
            reduction="none"
        )  # Don't reduce so we can apply weights

    def forward(self, predictions, targets, kp_index):
        # Extract kp from the last timestep and 9th feature of the input

        # Compute the weighting based on the extracted Kp input
        weighting = 1 + 99 * (torch.exp(kp_index) / torch.exp(torch.tensor(9.0)))

        # Compute MSE without reduction
        mse = self.mse_loss(predictions, targets)

        # Apply the weighting to the MSE
        weighted_mse = mse * weighting.unsqueeze(1)

        # Now reduce by taking the mean of the weighted MSE
        loss = torch.mean(weighted_mse)

        return loss


# # Example usage
# if __name__ == '__main__':
#     loss_fn = WeightedMSELoss()

#     # Simulate a batch of inputs with shape (batch_size, timesteps, features)
#     inputs = torch.rand(2, 91, 29)  # (batch_size=2, timesteps=91, features=29)

#     # Simulate predictions and targets
#     predictions = torch.rand(2, 1070)
#     targets = torch.rand(2, 1070)

#     # Compute the loss
#     loss, mse = loss_fn(predictions, targets, inputs)
#     print('Loss:', loss.item(), torch.mean(mse).item())
