import torch


def update_target_encoder(context_encoder, target_encoder, tau=0.996):
    """
    Update the target encoder using exponential moving average (EMA).
    """

    with torch.no_grad():

        for context_param, target_param in zip(
            context_encoder.parameters(),
            target_encoder.parameters()
        ):
            target_param.data.mul_(tau)
            target_param.data.add_(
                (1 - tau) * context_param.data
            )


def ijepa_loss(prediction, target):
    """
    Basic JEPA representation prediction loss.

    Prediction and target are expected to have the same shape.
    """

    target = target.detach()

    loss = torch.mean(
        torch.abs(prediction - target)
    )

    return loss