import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import torch.nn.init as init

#default_opt = {'alg': 'adam', 'lr': 1e-4, 'b1': 0.99, 'b2': 0.999,
#               'layers': 2, 'batch_size': 128}

class SubPixel1D(nn.Module):
    def __init__(self, r):
        super(SubPixel1D, self).__init__()
        self.r = r

    def forward(self, x):
        b, c, w = x.size()
        #print("check tensor", x.size())
        #input()
        x = x.view(b, c // self.r, self.r, w)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, c // self.r, w * self.r)

        return x

#def SubPixel1d(tensor, r): #(b,r,w)
#    ps = nn.PixelShuffle(r)
#    tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
#    tensor = ps(tensor)
#    #print(tensor.shape) #(b,1,w*r,r)
#    tensor = torch.mean(tensor, -1)
#    #print(tensor.shape) #(b,1,w*r)
#    return tensor
class AudioUNet(nn.Module):

    def __init__(self, layers = 4 ):

        super(AudioUNet, self).__init__()

        #self.n_filters = [128, 384, 512, 512, 512, 512, 512, 512]
        #self.n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9]

        self.n_filters = [128, 384, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9]

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.layers = layers

        # Downsampling layers
        for l, nf, fs in zip(list(range(self.layers)),
                             self.n_filters, self.n_filtersizes):

            conv_layer = nn.Conv1d(in_channels=1
            if len(self.downsampling_layers) == 0
                    else self.n_filters[len(self.downsampling_layers) - 1],
                              out_channels=nf,
                              kernel_size=fs,
                              stride=2,
                              padding=fs // 2)

            init.orthogonal_(conv_layer.weight)

            x = nn.Sequential(
                conv_layer,
                nn.BatchNorm1d(nf),
                nn.LeakyReLU(0.2)
            )

            self.downsampling_layers.append(x)

        # Bottleneck layer
        self.bottleneck_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.n_filters[-1],
                      out_channels=self.n_filters[-1],
                      kernel_size=self.n_filtersizes[-1],
                      stride=2,
                      padding=self.n_filtersizes[-1] // 2),
            nn.Dropout(0.5),
            nn.BatchNorm1d(self.n_filters[-1]),
            nn.LeakyReLU(0.2)
        )

        # Upsampling layers
        len_filters = len(self.n_filters)

        rev_n_filters = self.n_filters[::-1]
        rev_n_filtersizes = self.n_filtersizes[::-1]

        for ind in range(len_filters):

            nf = self.n_filters[-1]
            fs = rev_n_filtersizes[ind]

            if (ind == 0) :
                in_channels = nf
            elif (ind == 1) :
                in_channels = rev_n_filters[ind-1] + nf
            else :
                in_channels = rev_n_filters[ind-1] + rev_n_filters[ind-1]

            conv_layer = nn.Conv1d(in_channels = in_channels ,
                              out_channels = 2*rev_n_filters[ind],
                              kernel_size = fs,
                              stride = 1,
                              padding = fs // 2)

            init.orthogonal_(conv_layer.weight)

            x = nn.Sequential(
                    conv_layer,
                    nn.BatchNorm1d(2 * rev_n_filters[ind]),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    SubPixel1D(r=2)
                )

            self.upsampling_layers.append(x)

        conv_layer = nn.Conv1d(in_channels=2 * rev_n_filters[-1],
                  out_channels=2, kernel_size=9, padding=4)

        init.orthogonal_(conv_layer.weight)

        self.final_layer = nn.Sequential (
            conv_layer,
            SubPixel1D(r=2) )

    def forward(self, x):

        x = x.transpose(1,2)
        x_start = x
        downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            downsampling_l.append(x)

        x = self.bottleneck_layer(x)

        for l, l_in in list(zip( self.upsampling_layers, reversed(downsampling_l) )):

            x = torch.concat((l(x), l_in), axis=1)

        x = self.final_layer(x)

        #-----------------------------
        # Additive residual connection
        #-----------------------------
        x = x + x_start

        x = x.transpose(1,2)

        return x

    def create_objective(self, P, Y):
        # Compute L2 loss

        l2_loss = torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6
        norm = torch.mean(Y ** 2, dim=[1, 2])

        #print("*************************************")
        #print("l2_loss", l2_loss)

        avg_l2_loss = torch.mean(l2_loss, dim =0)

        avg_norm = torch.mean(norm, dim =0)

        sqrt_l2_loss = torch.sqrt(torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6)
        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        #print("sqrt_l2_loss", sqrt_l2_loss)
        #print("avg_sqrt_l2_loss", avg_sqrt_l2_loss)
        #print("*************************************")

        sqrn_l2_norm = torch.sqrt(torch.mean(Y ** 2, dim=[1, 2]))

        snr = 20 * torch.log10(sqrn_l2_norm / (sqrt_l2_loss + 1e-8))
        avg_snr = torch.mean(snr, dim=0)

        return avg_sqrt_l2_loss, avg_l2_loss, avg_norm, avg_snr


# Define the weights initialization function
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Example usage
if __name__ == "__main__":
    ## Initialize the model
    model = AudioUNet()
    model_state_dict = model.state_dict()
    model_state_dict_new = model.state_dict()

    for v, k in model_state_dict.items():
        #print(v, k.shape)
        ones_matrix = torch.ones_like(k)
        model_state_dict_new[v]  = ones_matrix

    model_state_dict =model_state_dict_new

    np_array_total = np.arange(0,1024)
    layers = 4

    np_array_total = np_array_total[:len(np_array_total) -
                                     (len(np_array_total) % (2 ** (layers + 1)))]

    input_tensor_total = torch.tensor(np_array_total.flatten(),
                                      dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    x_rnn = input_tensor_total.permute(0, 2, 1)
    print("size x_rnn", x_rnn.shape)

    x_rnn = input_tensor_total.permute(0, 1, 2)
    print("size x_rnn permute", x_rnn.shape)

    # Upscale the low-res version using the model
    with torch.no_grad():
        P = model(input_tensor_total).squeeze().numpy()

    output_tensor_total = P.flatten()

    np_array = np.array([[[-1.9767478, 0.06489189, -0.76302385, 1.6657038],
                 [-0.41181853, -0.1590211, -0.7415508, -0.17498407]],
                [[-0.8643993, -0.17599164, -1.35989, 0.16226906],
                 [0.25267616, -1.511163, 0.10261359, -0.60140926]]])

    input_tensor = torch.tensor(np_array, dtype = torch.float32)
    subpixel_layer = SubPixel1D(r = 2)
    output_tensor = subpixel_layer(input_tensor.permute(0,2,1))

    print("Input tensor shape:")
    print(input_tensor.shape)
    print("Output tensor shape:")
    print(output_tensor.shape)

    print("Input tensor:")
    print(input_tensor)
    print("Output tensor:")
    print(output_tensor)

    print("Input tensor total:")
    print(input_tensor_total)
    print("Output tensor total:")
    print(output_tensor_total)





