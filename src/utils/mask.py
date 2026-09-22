import torch


def random_mask(
    batch_size,
    num_patches,
    mask_ratio=0.25,
    device="cpu"
):
    """
    Create a random boolean mask for patch tokens.

    True  = masked
    False = visible
    """

    num_masked = int(
        num_patches * mask_ratio
    )

    masks = torch.zeros(
        batch_size,
        num_patches,
        dtype=torch.bool,
        device=device
    )

    for i in range(batch_size):

        indices = torch.randperm(
            num_patches,
            device=device
        )[:num_masked]

        masks[i, indices] = True

    return masks