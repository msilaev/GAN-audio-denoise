import torch.nn.init as init
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d
from torch.nn.utils import weight_norm, spectral_norm

###########################################
from .multiScaleConv import MultiscaleConv1DBlock as MultiscaleConvBlock
from .subPixel1D import SubPixel1D
from .superPixel1D import SuperPixel1D
###########################################

# Define the weights initialization function
def weights_init(m):

    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

#-------------------
#  Generator
#-------------------
class Generator(nn.Module):

    #def __init__(self, layers = 7, n_filters = (2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10)):
    def __init__(self, layers=4,  n_filters = (64, 128, 256, 512, 512)):
        #(128, 384, 512, 512)
        #(128, 128, 128, 128, 128, 256, 512))
        #128, 384, 512, 512, 512, 512, 512, 512
        #n_filters = (8, 16, 32, 64, 128, 128, 64, 32)

        super(Generator, self).__init__()

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.layers = layers
        self.n_filters = n_filters

        n_in = 1
        n_out_arr = []

        #-----------------
        # Downsampling layers
        #-----------------
        for l in range(self.layers):

            n_out = self.n_filters[l]//4

            conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                             out_channels = n_out)
            #conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.ReLU(),
                #nn.BatchNorm1d(4*n_in),
                #nn.LeakyReLU(0.2),
                SuperPixel1D(r=2)
            )

            self.downsampling_layers.append(x)

            # n_out comes from 4 stacked layers and SuperPixel

            #n_out = 8*n_in
            n_out_arr.append(8*n_out)
            n_in = 8*n_out

        # -----------------
        # Bottleneck layer
        # -----------------
        conv_layer_1 = MultiscaleConvBlock(in_channels=n_in,
                                         out_channels=n_in//8)

        conv_layer_2 = MultiscaleConvBlock(in_channels=n_in,
                                           out_channels=n_in//2)

        #conv_layer_1.apply(self.initialize_weights)
        #conv_layer_2.apply(self.initialize_weights)

        x = nn.Sequential(
            conv_layer_1,
            nn.ReLU(),
            #nn.BatchNorm1d(4 * n_in),
            #nn.LeakyReLU(0.2),
            SuperPixel1D(r=2),
            conv_layer_2,
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            SubPixel1D(r=2))

        self.bottleneck_layer = x
        # we add here also stack
        n_out =  n_in

        # -----------------
        # Upsampling layer
        # -----------------
        for l in range(self.layers):

            n_in = n_out + n_out_arr[len(n_out_arr) - l - 1]

            n_out_conv = self.n_filters[len(n_out_arr) - l - 1] // 4

            #print("upsampling n_in", n_in)

            conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                             out_channels = n_out_conv)

            #conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SubPixel1D(r=2))

            self.upsampling_layers.append(x)

            # n_out comes from Skip Connection, 4 stacked layers, SubPixel
            n_out = 2*n_out_conv

        ####################
        # define final layer
        ####################
        x = nn.Conv1d(n_out, 1, kernel_size=27, padding=13)
        self.final_layer = x

        #self.final_layer = nn.Conv1d(in_channels=n_out,
        #          out_channels=1, kernel_size=9, padding=4)

        #init.orthogonal_(conv_layer.weight)

        #self.final_layer = nn.Sequential (
        #    conv_layer,
        #    SubPixel1D(r=2) )
        
        #x.apply(self.initialize_weights)
        

        # Apply custom weight initialization
        #self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, x):

        x = x.transpose(1,2)
        x_start = x
        downsampling_l = []

        for layer in self.downsampling_layers:

            x = layer(x)
            downsampling_l.append(x)
        x = self.bottleneck_layer(x)

        for l, l_in in list(zip(self.upsampling_layers, reversed(downsampling_l) )):

            x = torch.concat((x, l_in), axis=1)
            x = l(x)

        x = self.final_layer(x)
        x = x + x_start
        x = x.transpose(2,1)

        return x

    def create_objective(self, P, Y):
        # Compute L2 loss

        l2_loss = torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6

        norm = torch.mean(Y ** 2, dim=[1, 2])

        avg_l2_loss = torch.mean(l2_loss, dim =0)
        avg_norm = torch.mean(norm, dim =0)

        sqrt_l2_loss = torch.sqrt(torch.mean((P - Y) ** 2, dim=[1, 2]) + 1e-6)
        avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        #avg_sqrt_l2_loss = torch.mean(sqrt_l2_loss, dim=0)

        sqrn_l2_norm = torch.sqrt(torch.mean(Y ** 2, dim=[1, 2]))

        snr = 20 * torch.log10(sqrn_l2_norm / (sqrt_l2_loss + 1e-8))
        avg_snr = torch.mean(snr, dim=0)

        return avg_sqrt_l2_loss, avg_l2_loss, avg_norm, avg_snr

#-------------------
#  Discriminator
#-------------------
class Discriminator(nn.Module):
    #MultiscaleConvBlockLeaky

    #def __init__(self, layers, time_dim, n_filters = ( 32,  64, 64, 128, 128, 256, 256  )):
    def __init__(self, layers, time_dim, n_filters = (64, 128, 256, 256, 512)):
        #                             n_filters = (64, 128, 256, 256)

        super(Discriminator, self).__init__()

        self.layers = layers
        self.downsampling_layers = nn.ModuleList()

        n_in  = 1
        n_out = 128

        conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                         out_channels = n_out//4)

        #conv_layer.apply(self.initialize_weights)

        n_in = n_out

        x = nn.Sequential(
            conv_layer,
            nn.LeakyReLU(0.2))

        #------------
        # 32 channels
        #------------
        self.downsampling_layers.append(x)

        self.n_filters = n_filters

        for l in range(self.layers):

            n_out = self.n_filters[l]//4

            #print("n_in, n_out", n_in, n_out)
            #input()

            conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                        out_channels = n_out)
            #conv_layer.apply(self.initialize_weights)
            batch_norm = nn.BatchNorm1d(4*n_out)
            #batch_norm.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                batch_norm,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SuperPixel1D(r=2))

            #x.apply(self.initialize_weights)

            n_in = 8*n_out
            # ------------
            # 32 - 64 - 128 - 256 - 256 - 128 - 64 - 32 channels, 2 x n_filters
            # ------------

            self.downsampling_layers.append(x)

        self.input_features = n_in*time_dim // (2 ** self.layers)

        #print("dimension", n_in, time_dim, (2 ** self.layers), self.input_features)

        fc_outdim = 1024//32

        self.fc_1 = nn.Linear(self.input_features, fc_outdim)
        self.fc_2 = nn.Linear(fc_outdim, 1)

        self.final_layer = nn.Sequential(
            self.fc_1,
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            self.fc_2)
            #nn.Sigmoid())

        # Apply custom weight initialization
        #self.apply(self.initialize_weights)

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, x):

        fmap_r = []

        x = x.transpose(1,2)

        for l in self.downsampling_layers:
            #print("x shape", x.shape)

            x = l(x)
            fmap_r.append(x)

        x = x.view(x.size(0), -1)

        #print("x shape before final layer", x.shape)
        x = self.final_layer(x)
        #y = self.final_layer(y)

        #x = l(x)
        #y = l(y)

        #fmap_r.append(x)
        #fmap_g.append(y)

        return x, fmap_r

# ------------
# Other discriminator
# ------------
class AudioDiscriminator(nn.Module):
    def __init__(self):
        super(AudioDiscriminator, self).__init__()

        # Define the 12 blocks
        self.blocks = nn.ModuleList()

        in_channels = 1  # Assuming audio input has one channel
        for i in range(5):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, in_channels * 2, kernel_size=31, stride=2, padding=15),
                nn.BatchNorm1d(in_channels * 2),
                nn.LeakyReLU(0.3)
            ))
            in_channels *= 2

        # Final block
        self.blocks.append(nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=31, stride=2, padding=15),
            nn.LeakyReLU(0.3)
        ))

        # Fully connected layer to compress the output to a single value
        self.fc =  nn.Linear(128, 1)


    def initialize_weights(self, module):
        if isinstance(module, nn.Conv1d):
            init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)

        # Reshape for the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

# -------------------
#  AutoEncoder
# -------------------
class AutoEncoder(nn.Module):

    #def __init__(self, layers=3, n_filters=(2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10)):

    def __init__(self, layers=4, n_filters = (64, 128, 256, 256)):
        # 128, 384, 512, 512, 512, 512, 512, 512
        # n_filters = (8, 16, 32, 64, 128, 128, 64, 32)

        super(AutoEncoder, self).__init__()

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.layers = layers
        self.n_filters = n_filters

        n_in = 1
        n_out_arr = []

        # -----------------
        # Downsampling layers
        # -----------------
        for l in range(self.layers):

            n_out = self.n_filters[l] // 4

            conv_layer = MultiscaleConvBlock(in_channels=n_in,
                                                 out_channels=n_out)

            x = nn.Sequential(
                conv_layer,
                nn.ReLU(),
                SuperPixel1D(r=2)
                )

            self.downsampling_layers.append(x)

            n_out_arr.append(8 * n_out)
            n_in = 8 * n_out

        # -----------------
        # Bottleneck layer
        # -----------------
        conv_layer_1 = MultiscaleConvBlock(in_channels=n_in,
                                               out_channels=n_in // 8)

        conv_layer_2 = MultiscaleConvBlock(in_channels=n_in,
                                               out_channels=n_in // 2)

        x = nn.Sequential(
                conv_layer_1,
                nn.ReLU(),
                SuperPixel1D(r=2),
                conv_layer_2,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SubPixel1D(r=2))

        self.bottleneck_layer = x
        n_out = n_in

        # -----------------
        # Upsampling layer
        # -----------------
        for l in range(self.layers):

            n_out_conv = self.n_filters[len(n_out_arr) - l - 1] // 4

            conv_layer = MultiscaleConvBlock(in_channels=n_in,
                                                 out_channels=n_out_conv)

            x = nn.Sequential(
                    conv_layer,
                    nn.Dropout(0.5),
                    nn.LeakyReLU(0.2),
                    SubPixel1D(r=2))

            self.upsampling_layers.append(x)
            n_out = 2 * n_out_conv
            n_in = n_out

        x = nn.Conv1d(n_out, 1, kernel_size=27, padding=13)
        self.final_layer = x
        self.apply(self.initialize_weights)

    def forward(self, x):

        x = x.transpose(1, 2)
        #downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            #downsampling_l.append(x)
        x = self.bottleneck_layer(x)

        for l in self.upsampling_layers:
            x = l(x)

        x = self.final_layer(x)
        x = x.transpose(2, 1)

        return x

    def get_features(self, x):

        x = x.transpose(1, 2)
        #downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            #downsampling_l.append(x)
        x = self.bottleneck_layer(x)

        return x

    def initialize_weights(self, module):

        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def create_objective(self, x, y):
        # Compute L2 loss

        l2_loss = torch.mean((x - y) ** 2, dim=[1, 2]) + 1e-6
        avg_l2_loss = torch.mean(l2_loss, dim=0)

        l2_norm_x = torch.mean(x ** 2, dim=[1, 2]) + 1e-6
        avg_l2_norm_x = torch.mean(l2_norm_x, dim=0)

        l2_norm_y = torch.mean(y ** 2, dim=[1, 2]) + 1e-6
        avg_l2_norm_y = torch.mean(l2_norm_y, dim=0)

        return avg_l2_loss, avg_l2_norm_x, avg_l2_norm_y

########################################################
class MelganDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MelganDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding="same")),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding="same"))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        # Global average pooling???

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_random_window=False, reversed_discs=False):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            MelganDiscriminator(),
            MelganDiscriminator(),
            MelganDiscriminator(),
            MelganDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2),
            AvgPool1d(4, 2),
            AvgPool1d(4, 2)
        ])

    def forward(self, y, y_hat):


        y = y.transpose(2, 1)
        y_hat = y_hat.transpose(2, 1)
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    #r_losses = []
    #g_losses = []

    r_losses = 0
    g_losses = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):

        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)

        r_losses += r_loss
        g_losses += g_loss

        loss += (r_loss + g_loss)

        #print(torch.mean(dr).item(), torch.mean(dg).item())
        #print( r_loss.item(), g_loss.item(), (r_loss.item() + g_loss.item()))

        #r_losses.append(r_loss.item())
        #g_losses.append(g_loss.item())

    return  loss , r_losses , g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss


def BCEWithSquareLoss(discriminator_output, targets):
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    loss = bce_loss(discriminator_output, targets)
    
    #mse_loss = nn.MSELoss()
    #loss = mse_loss(discriminator_output, targets)
    
    return loss
##########################################################
# Example usage
# if __name__ == "__main__":
#    # Define input tensor with shape (batch_size, in_channels, height, width)
#    input_tensor = torch.randn(1, 1, 100, 10)  # Example input

#    # Initialize the multiscale block
#    block = MultiscaleConvBlock(in_channels=1, out_channels=16)

#    # Forward pass
#    output = block(input_tensor)

#    print(output.shape)
