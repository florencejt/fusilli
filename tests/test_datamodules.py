import pytest

from fusionlibrary.datamodules import downsample_img_batch
import torch


def test_downsample_img_batch():
    """
    Tests the downsample_img_batch function.

    Returns
    -------
    None
    """
    # Arrange
    num_channels = 3
    img_height = 10
    img_width = 10
    img_batch = torch.rand(num_channels, img_height, img_width)

    # Act
    downsampled_img_batch = downsample_img_batch(img_batch, (5, 5))

    # Assert
    assert downsampled_img_batch.shape == (num_channels, 5, 5)
