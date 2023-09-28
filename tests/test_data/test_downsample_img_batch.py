import pytest
import torch
import torch.nn.functional as F
from fusilli.data import (
    downsample_img_batch,
)  # Import the function from your module


def test_downsample_img_batch_no_downsampling():
    # Test when output_size is None, no downsampling should be performed
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = None

    downsampled_img = downsample_img_batch(imgs, output_size)

    assert (
        downsampled_img.shape == imgs.shape
    )  # Output shape should be the same as input


def test_downsample_img_batch_2D_image():
    # Test with a 2D image and valid output_size
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = (64, 64)  # Valid output_size for 2D image

    downsampled_img = downsample_img_batch(imgs, output_size)

    expected_shape = (16, 3, 64, 64)
    assert (
        downsampled_img.shape == expected_shape
    )  # Output shape should match the specified size


def test_downsample_img_batch_3D_image():
    # Test with a 3D image and valid output_size
    imgs = torch.randn(16, 3, 128, 128, 128)  # Batch of 3-channel 3D images
    output_size = (64, 64, 64)  # Valid output_size for 3D image

    downsampled_img = downsample_img_batch(imgs, output_size)

    expected_shape = (16, 3, 64, 64, 64)
    assert (
        downsampled_img.shape == expected_shape
    )  # Output shape should match the specified size


def test_downsample_img_batch_negative_output_size():
    # Test with negative output_size, should raise ValueError
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = (-1, 64)  # Negative output_size

    with pytest.raises(ValueError):
        downsample_img_batch(imgs, output_size)  # Function should raise a ValueError


def test_downsample_img_batch_output_size_larger_than_image():
    # Test with output_size larger than image dimensions, should raise ValueError
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = (256, 256)  # Larger output_size

    with pytest.raises(ValueError):
        downsample_img_batch(imgs, output_size)  # Function should raise a ValueError


def test_downsample_img_batch_output_size_wrong_dimension():
    # Test with output_size having more dimensions than the image, should raise ValueError
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = (100, 100, 100)  # Larger output_size dimensions

    with pytest.raises(ValueError):
        downsample_img_batch(imgs, output_size)  # Function should raise a ValueError


def test_downsample_img_batch_invalid_output_size_for_2D_image():
    # Test with invalid output_size for a 2D image, should raise ValueError
    imgs = torch.randn(16, 3, 128, 128)  # Batch of 3-channel images
    output_size = (64, 64, 64)  # Invalid output_size for a 2D image

    with pytest.raises(ValueError):
        downsample_img_batch(imgs, output_size)  # Function should raise a ValueError


def test_downsample_img_batch_invalid_output_size_for_3D_image():
    # Test with invalid output_size for a 3D image, should raise ValueError
    imgs = torch.randn(16, 3, 128, 128, 128)  # Batch of 3-channel 3D images
    output_size = (64, 64)  # Invalid output_size for a 3D image

    with pytest.raises(ValueError):
        downsample_img_batch(imgs, output_size)  # Function should raise a ValueError


# Run pytest
if __name__ == "__main__":
    pytest.main()
