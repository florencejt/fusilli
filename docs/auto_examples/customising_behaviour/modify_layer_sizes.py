"""
How to modify architectures of fusion models
############################################

This tutorial will show you how to modify the architectures of fusion models.

Notes
-----

- For image channel-wise attention, the mod1 layers and the img layers must have the same number of layers and the same
    number of output channels in each layer. The kernel size and padding etc can be different.
- For tabular channel-wise attention, the mod1 layers and the mod2 layers must have the same number of layers and the
    same number of output features in each layer.
"""
