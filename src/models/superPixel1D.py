import torch
import torch.nn as nn

class SuperPixel1D(nn.Module):

    def __init__(self, r):
        super(SuperPixel1D, self).__init__()
        self.r = r

    def forward(self, x):
        b, c, w = x.size()
        if w % self.r != 0:
            raise ValueError(f"Width of input tensor "
                             f"{self.w} must be divisible by r {self.r}")

        #print("check tensor", x.size())
        #input()

        x = x.view(b, c, w // self.r, self.r)
        #x = x.view(b, c // self.r, self.r, w)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, c * self.r, w // self.r)

        return x