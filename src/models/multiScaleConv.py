import torch
import torch.nn as nn
import torch.nn.init as init

from .superPixel1D import SuperPixel1D
from .subPixel1D import SubPixel1D
from torch.nn.utils import weight_norm, spectral_norm

class MultiscaleConv1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(MultiscaleConv1DBlock, self).__init__()

        # Define parallel convolutional layers with different kernel sizes
        self.conv3 = (nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv9 = (nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4))
        self.conv27 = (nn.Conv1d(in_channels, out_channels, kernel_size=27, padding=13))
        self.conv81 = (nn.Conv1d(in_channels, out_channels, kernel_size=81, padding=40))

        # Batch normalization after concatenation
        self.bn = nn.BatchNorm1d(out_channels * 4)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the convolutional layers
        out3 = self.conv3(x)
        out9 = self.conv9(x)
        out27 = self.conv27(x)
        out81 = self.conv81(x)

        # Concatenate the outputs along the channel dimension
        out = torch.cat([out3, out9, out27, out81], dim=1)

        return out

# Example usage
if __name__ == "__main__":
    # Define input tensor with shape (batch_size, in_channels, sequence_length)
    input_tensor = torch.randn(10, 1, 100)
    # Example input with batch_size=10, in_channels=1, sequence_length=100

    # Initialize the multiscale block
    block = MultiscaleConv1DBlock(in_channels=1, out_channels=1)

    #block_1 = MultiscaleConv1DBlock(in_channels=8, out_channels=8)

    superPixel = SuperPixel1D(r=2)
    subPixel = SubPixel1D(r=2)

    # Forward pass
    output =   superPixel(block(input_tensor))
    output_1 = subPixel(block(input_tensor))

    print( input_tensor.shape, (block(input_tensor)).shape, output.shape, output_1.shape)
    # Expected shape: (10, 64, 100) because out_channels * 4 = 16 * 4 = 64
    #output = block_1(superPixel(output))

    #print(output.shape)  # Expected shape: (10, 64, 100) because out_channels * 4 = 16 * 4 = 64
