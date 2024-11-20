import torch
import torch.nn as nn
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

from dataset_batch import BatchData

#import torch.nn.functional as F
#from torchinfo import summary
import numpy as np
#from scipy import interpolate

import torch.nn.init as init
#from multiScaleConv import MultiscaleConv1DBlock as MultiscaleConvBlock
#from subPixel1D import SubPixel1D
#from superPixel1D import SuperPixel1D

###########################################
from .multiScaleConv import MultiscaleConv1DBlock as MultiscaleConvBlock
from .subPixel1D import SubPixel1D
from .superPixel1D import SuperPixel1D
###########################################

# Define the weights initialization function
def load_h5(h5_path):
  # load training data
  print(h5_path)
  with h5py.File(h5_path, 'r') as hf:
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))

  return X, Y

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

    def __init__(self, layers = 7, n_filters = (2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10)):
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
            conv_layer.apply(self.initialize_weights)

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

        conv_layer_1.apply(self.initialize_weights)
        conv_layer_2.apply(self.initialize_weights)

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

            conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SubPixel1D(r=2))

            self.upsampling_layers.append(x)
            #print(len(n_out_arr), l)

            # n_out comes from Skip Connection, 4 stacked layers, SubPixel
            #n_out = n_in//2
            n_out = 2*n_out_conv

            #n_in = n_out

        ####################
        # define final layer
        ####################
        x = nn.Conv1d(n_out, 1, kernel_size=27, padding=13)
        
        x.apply(self.initialize_weights)
        
        self.final_layer = x

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

            #print("downsampling", x.shape)

        x = self.bottleneck_layer(x)

        #print("after bottleneck", x.shape)

        for l, l_in in list(zip(self.upsampling_layers, reversed(downsampling_l) )):

            #print(x.shape, l_in.shape)

            x = torch.concat((x, l_in), axis=1)

            #print("upsampling stak", x.shape)

            x = l(x)

            #print("upsampling", x.shape)

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

    def __init__(self, layers, time_dim, n_filters = ( 16, 32, 64, 128, 256, 512, 1024 )):

        super(Discriminator, self).__init__()

        self.layers = layers
        self.downsampling_layers = nn.ModuleList()

        n_in  = 1
        n_out = 8

        conv_layer = MultiscaleConvBlock(in_channels = n_in,
                                         out_channels = n_out//4)

        conv_layer.apply(self.initialize_weights)

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
            conv_layer.apply(self.initialize_weights)
            batch_norm = nn.BatchNorm1d(4*n_out)
            batch_norm.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                batch_norm,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SuperPixel1D(r=2))

            x.apply(self.initialize_weights)

            n_in = 8*n_out
            # ------------
            # 32 - 64 - 128 - 256 - 256 - 128 - 64 - 32 channels, 2 x n_filters
            # ------------

            self.downsampling_layers.append(x)

        self.input_features = n_in*time_dim // (2 ** self.layers)

        #print("dimension", n_in, time_dim, (2 ** self.layers), self.input_features)

        fc_outdim = 128

        self.fc_1 = nn.Linear(self.input_features, fc_outdim)
        self.fc_2 = nn.Linear(fc_outdim, 1)

        self.final_layer = nn.Sequential(
            self.fc_1,
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            self.fc_2,
            nn.Sigmoid())

        # Apply custom weight initialization
        self.apply(self.initialize_weights)

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

        for l in self.downsampling_layers:
            #print("x shape", x.shape)

            x = l(x)

        x = x.view(x.size(0), -1)

        #print("x shape before final layer", x.shape)
        x = self.final_layer(x)

        return x


# -------------------
#  AutoEncoder
# -------------------

class AutoEncoder(nn.Module):

    def __init__(self, layers=7, n_filters=(2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10)):
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
            conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                conv_layer,
                nn.ReLU(),
                # nn.BatchNorm1d(4*n_in),
                # nn.LeakyReLU(0.2),
                SuperPixel1D(r=2)
                )

            self.downsampling_layers.append(x)

            # n_out comes from 4 stacked layers and SuperPixel

            # n_out = 8*n_in
            n_out_arr.append(8 * n_out)
            n_in = 8 * n_out

            #print(n_in)

        # -----------------
        # Bottleneck layer
        # -----------------
        conv_layer_1 = MultiscaleConvBlock(in_channels=n_in,
                                               out_channels=n_in // 8)

        conv_layer_2 = MultiscaleConvBlock(in_channels=n_in,
                                               out_channels=n_in // 2)

        conv_layer_1.apply(self.initialize_weights)
        conv_layer_2.apply(self.initialize_weights)

        x = nn.Sequential(
                conv_layer_1,
                nn.ReLU(),
                # nn.BatchNorm1d(4 * n_in),
                # nn.LeakyReLU(0.2),
                SuperPixel1D(r=2),
                conv_layer_2,
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                SubPixel1D(r=2))

        self.bottleneck_layer = x
        # we add here also stack
        n_out = n_in

        # -----------------
        # Upsampling layer
        # -----------------
        for l in range(self.layers):
            #n_in = n_out + n_out_arr[len(n_out_arr) - l - 1]

            n_out_conv = self.n_filters[len(n_out_arr) - l - 1] // 4

            # print("upsampling n_in", n_in)

            conv_layer = MultiscaleConvBlock(in_channels=n_in,
                                                 out_channels=n_out_conv)

            conv_layer.apply(self.initialize_weights)

            x = nn.Sequential(
                    conv_layer,
                    nn.Dropout(0.5),
                    nn.LeakyReLU(0.2),
                    SubPixel1D(r=2))

            self.upsampling_layers.append(x)
            # print(len(n_out_arr), l)
            # n_out comes from Skip Connection, 4 stacked layers, SubPixel
            # n_out = n_in//2
            n_out = 2 * n_out_conv

            n_in = n_out

        ####################
        # define final layer
        ####################
        x = nn.Conv1d(n_out, 1, kernel_size=27, padding=13)

        x.apply(self.initialize_weights)

        self.final_layer = x

        # Apply custom weight initialization
        # self.apply(self.initialize_weights)

    def initialize_weights(self, module):

        if isinstance(module, nn.Conv1d):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm1d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def get_features(self, x):

        x = x.transpose(1, 2)
        downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            downsampling_l.append(x)

        x = self.bottleneck_layer(x)

        return x

    def forward(self, x):

        x = x.transpose(1, 2)

        downsampling_l = []

        for layer in self.downsampling_layers:
            x = layer(x)
            downsampling_l.append(x)

            #print(x.shape)

        x = self.bottleneck_layer(x)
        #print(x.shape)

        for l, l_in in list(zip(self.upsampling_layers, reversed(downsampling_l))):

            x = l(x)
            #print(x.shape)

        x = self.final_layer(x)

        x = x.transpose(2, 1)

        return x

    def create_objective(self, x, y):
        # Compute L2 loss

        l2_loss = torch.mean((x - y) ** 2, dim=[1, 2]) + 1e-6
        avg_l2_loss = torch.mean(l2_loss, dim=0)

        l2_norm_x = torch.mean(x ** 2, dim=[1, 2]) + 1e-6
        avg_l2_norm_x = torch.mean(l2_norm_x, dim=0)

        l2_norm_y = torch.mean(y ** 2, dim=[1, 2]) + 1e-6
        avg_l2_norm_y = torch.mean(l2_norm_y, dim=0)

        return avg_l2_loss, avg_l2_norm_x, avg_l2_norm_y

def train_gan():

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    adversarial_loss = torch.nn.BCELoss()

    #--------
    # learning parameters
    #--------
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    batch_size = 16
    epochs = 50
    time_dim = 8192

    #--------
    # real and fake labels
    #--------
    real_label = 1
    fake_label = 0

    # --------
    # Initialize the model
    # --------
    num_layers_disc = 1
    discriminator = Discriminator(layers=num_layers_disc, time_dim=time_dim).to(device)

    #output = model_discrim(input_tensor)
    #print("Discriminator", input_tensor.shape, output.shape)

    num_layers_gen = 1
    generator = Generator(layers=num_layers_gen).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    fixed_noise = torch.randn(batch_size, time_dim, 1, device=device)

    # --------
    # define data loader
    # --------
    base_dir = Path("..") / ".."/ "data"
    train_path = base_dir / "vctk" / "speaker1" / "vctk-speaker1-train.3.16000.8192.4096.h5"
    X_train, Y_train = load_h5(train_path)

    train_loader = torch.utils.data.DataLoader(BatchData(X_train, Y_train),
                                               batch_size=batch_size,
                                               shuffle=True, drop_last = True)
    # --------
    # check untrained generator output
    # --------
    #real_batch = next(iter(train_loader))
    #real_feature = real_batch[0]
    #real_labels = real_batch[1]
    #print("batch shape", real_feature.shape)

    #with torch.no_grad():
    #    fake = generator(fixed_noise).detach().cpu()
    #print("fake shape", fake.shape)

    # --------
    # train the model
    # --------
    d_loss_arr=[]
    g_loss_arr=[]

    for epoch in range(1, epochs + 1):

        for i, (lr_sound, target_sound) in enumerate(train_loader, 0):

            print("sound shape", lr_sound.shape)

            real_sound = target_sound.to(device)
            b_size = real_sound.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # ------------
            # Train Discriminator
            # ------------
            discriminator.zero_grad()
            real_output = discriminator(real_sound).view(-1)

            real_loss = adversarial_loss(real_output, label)
            real_loss.backward()
            D_x = real_output.mean().item()

            noise = lr_sound.to(device)
            fake_sound = generator(noise).to(device)
            label.fill_(fake_label)

            output = discriminator(fake_sound.detach()).view(-1)

            fake_loss = adversarial_loss(output, label)
            fake_loss.backward()
            D_G_z1 = output.mean().item()

            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            # optimizer_G.zero_grad()
            generator.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_sound).view(-1)

            loss, avg_l2_loss, avg_norm, avg_snr = \
                generator.create_objective(fake_sound, real_sound)

            g_loss = loss+ 0.001*adversarial_loss(fake_output, label)

            g_loss.backward()
            D_G_z2 = fake_output.mean().item()
            optimizer_G.step()

            d_loss_arr.append(d_loss.item())
            g_loss_arr.append(g_loss.item())

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(train_loader),
                     d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

    d_loss_arr = np.array(d_loss_arr)
    g_loss_arr = np.array(g_loss_arr)

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss_arr, label="G")
    plt.plot(d_loss_arr, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("loss.png")
    plt.show()

# Example usage
if __name__ == "__main__":

    #train_gan()

    if torch.cuda.is_available():
        print("CUDA!")
        device = torch.device('cuda')
    else:
        print("NO CUDA!")
        device = torch.device('cpu')

    ## Initialize the model
    num_layers_disc = 7
    input_tensor = torch.randn(10, 8192, 1)

    model_discrim = Discriminator(layers = num_layers_disc,
                                  time_dim = input_tensor.shape[1])

    output = model_discrim(input_tensor)

    print("Discriminator", input_tensor.shape, output.shape)

    num_layers_gen = 7
    model_gen = Generator(layers = num_layers_gen)

    #print(model)
    # Calculate the number of trainable parameters
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print(p for p in model.parameters() if p.requires_grad)
    #for p in model.parameters():
    #    if p.requires_grad:
    #        print(p, p.numel)

    #print(f"Number of trainable parameters: {trainable_params}")

    #model.layers = 1
    output = model_gen(input_tensor)

    print("Generator", input_tensor.shape, output.shape)

    adversarial_loss = torch.nn.BCELoss()

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
