import math

import torch


def encode(tensor: torch.Tensor, n_bins: int):
    # Flatten tensor
    flat_tensor = tensor.flatten()

    # Sort the flattened tensor
    sorted_tensor, _ = torch.sort(flat_tensor)

    # Determine the thresholds for each bin
    n_points_per_bin = int(math.ceil(len(sorted_tensor) / n_bins))
    groups = torch.split(sorted_tensor, n_points_per_bin)
    centroids = torch.stack([group.float().mean() for group in groups])

    # Create thresholds
    thresholds = torch.tensor([group[0] for group in groups])
    # inf = torch.tensor([float("inf")])
    # thresholds = torch.cat([thresholds, inf])

    # Assign each value in the flattened tensor to a bucket
    # The bucket number is the quantized value
    quantized_tensor = torch.bucketize(flat_tensor, thresholds)
    quantized_tensor = torch.clamp(quantized_tensor, 0, len(centroids) - 1)

    # Reshape the quantized tensor to the original tensor's shape
    quantized_tensor = quantized_tensor.view(tensor.shape)

    return quantized_tensor, centroids, thresholds


def check(x: torch.Tensor, n_bins: int):
    y, centroids, _ = encode(x, n_bins)
    z = centroids[y]
    diffs = (x - z).abs()
    return diffs.mean().item(), diffs.max().item()


if __name__ == "__main__":
    from utils import set_seed

    set_seed(0)

    for n_bins in range(3, 10):
        for max_x in range(2, n_bins):
            x = torch.randint(1, max_x + 1, (10, 10))
            x = torch.log2(x)
            norm_1, norm_inf = [round(x, 2) for x in check(x, n_bins)]
            print(f"Bins: {n_bins}, Max: {max_x}, Norm 1: {norm_1}, Norm ∞: {norm_inf}")

    for n_bins in range(2, 10):
        for max_x in range(n_bins, n_bins * 2):
            x = torch.randint(1, n_bins, (10, 10))
            x = torch.log2(x)
            norm_1, norm_inf = [round(x, 2) for x in check(x, n_bins)]
            print(f"Bins: {n_bins}, Max: {max_x}, Norm 1: {norm_1}, Norm ∞: {norm_inf}")
