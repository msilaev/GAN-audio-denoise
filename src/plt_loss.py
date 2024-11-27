import matplotlib.pyplot as plt
import numpy as np
import argparse
from models.io import load_h5

# -------------------
# parser
# -------------------
def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--sr')

    return parser

def plt_loss_evolution(file_name_logs_val, out_dir_name):

    g_gan_loss_train = []
    d_loss_train = []

    #print(file_name_logs_val)

    with  open(file_name_logs_val + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_train.append( float(x[1]) )

            g_gan_loss_train.append(float(x[5]))

    d_loss_train = np.array(d_loss_train)
    g_gan_loss_train = np.array(g_gan_loss_train)

    # Create a window of size
    window_size = 1
    window = np.ones(window_size) / window_size

    # Compute the moving average using np.convolve
    g_gan_loss_train = np.convolve(g_gan_loss_train, window, mode='valid')
    d_loss_train = np.convolve(d_loss_train, window, mode='valid')

    # Set font sizes globally
    plt.figure(figsize=(5, 5))

    plt.rcParams.update({'font.size': 25})  # General font size
    plt.rcParams.update({'axes.titlesize': 25})  # Title font size
    plt.rcParams.update({'axes.labelsize': 25})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 25})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 25})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 25})  # Y tick label font size

    x_label = "Iteration, $10^3$"
    plt.ylabel("<Adv Loss>$_{1}$")

    plt.plot(np.arange( g_gan_loss_train.shape[0])/1000, g_gan_loss_train, label='Gen', marker = "o")
    plt.plot(np.arange( d_loss_train.shape[0])/1000, d_loss_train, label='Discr', marker = "+")

    plt.xlabel(x_label)
    plt.ylim([0,2])

    plt.xlim([0, 100])
    plt.xticks([0, 100])

    plt.legend()
    plt.grid()
    plt.tight_layout()

    figure_file_name =  out_dir_name + "evolution.png"
    print(figure_file_name)

    #print(figure_file_name)
    #plt.show()

    plt.savefig(figure_file_name, format='png')
    input()

def plt_loss_epoch(file_name_logs_val, file_name_logs_train, out_dir_name, args):

    g_loss_val = []
    g_gan_loss_val = []
    d_loss_val = []
    mse_loss_val = []
    snr_val =[]
    lsd_val = []

    g_loss_train = []
    g_gan_loss_train = []
    d_loss_train = []
    mse_loss_train = []

    snr_train =[]
    lsd_train = []

    with  open(file_name_logs_val + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_val.append(float(x[1]) + float(x[2]))

            g_loss_val.append(float(x[3]))
            g_gan_loss_val.append(float(x[4]))
            mse_loss_val.append(float(x[5]))

            snr_val.append(float(x[8]))
            lsd_val.append(float(x[7]))

    with  open(file_name_logs_train + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_train.append(float(x[1]) + float(x[2]))

            g_loss_train.append(float(x[3]))
            g_gan_loss_train.append(float(x[4]))
            mse_loss_train.append(float(x[5]))

            snr_train.append(float(x[8]))
            lsd_train.append(float(x[7]))

    g_gan_loss_val = np.array(g_gan_loss_val)
    mse_loss_val = np.array(mse_loss_val)
    d_loss_val = np.array(d_loss_val)
    lsd_val = np.array(lsd_val)
    snr_val = np.array(snr_val)

    mse_loss_train = np.array(mse_loss_train)

    lsd_train = np.array(lsd_train)
    snr_train = np.array(snr_train)

    plt.rcParams.update({'font.size': 25})  # General font size
    plt.rcParams.update({'axes.titlesize': 25})  # Title font size
    plt.rcParams.update({'axes.labelsize': 25})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 25})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 25})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 25})  # Y tick label font size

    x_label = "Epoch, $10^2$"
    plt.xticks([0, 1, 2])
    plt.xlim([0, 2])

    file_name =  args.model + "_SNR_loss"

    plt.figure()
    plt.figure(figsize=(5, 5))

    plt.plot(np.arange( snr_val.shape[0])/100, snr_val, label ='SNR val', marker="o")
    plt.plot(np.arange( snr_train.shape[0])/100, snr_train, label = 'SNR train', marker="+")
    plt.xlabel(x_label)

    plt.ylabel("SNR")
    plt.ylim([0,35])
    plt.xticks([0, 1, 2])
    plt.xlim([0, 2])
    #if args.speaker == "multi":
    #    plt.xlim([0, 5])
    #elif args.speaker == "single":
    #    plt.xlim([0, 5])
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #                  "sr=" + str(int(args.sr)//1000) + " KHz")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    figure_file_name =  out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ################### LSD #####################
    file_name = args.model + "_LSD_loss"

    plt.figure()
    plt.figure(figsize=(5, 5))

    plt.plot(np.arange( lsd_val.shape[0])/100, lsd_val, label= 'LSD val', marker="o")
    plt.plot(np.arange( lsd_train.shape[0])/100, lsd_train, label= 'LSD train', marker="+")
    plt.xlabel(x_label)
    plt.ylabel("LSD")
    plt.ylim([0, 8])
    plt.xticks([0, 1, 2])
    plt.xlim([0, 2])
    #if args.speaker == "multi":
    #    plt.xlim([0, 5])
    #elif args.speaker == "single":
    #    plt.xlim([0, 5])

    plt.legend()
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #          "sr=" + str(args.sr) + " Hz")
    plt.grid()
    plt.tight_layout()

    figure_file_name = out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ############## MSE #####################
    file_name = args.model + "_MSE_loss"
    plt.figure()
    plt.figure(figsize=(5, 5))

    plt.plot(np.arange( mse_loss_val.shape[0])/100, mse_loss_val , label='MSE val', marker="o")
    plt.plot(np.arange( mse_loss_train.shape[0])/100, mse_loss_train, label='MSE train', marker="+")
    plt.xlabel(x_label)
    plt.ylabel("MSE")
    #plt.ylim([0, 0.00005])

    plt.legend()
    plt.xticks([0, 1, 2])
    plt.xlim([0, 2])
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #          "sr=" + str(int(args.sr)//1000) + " KHz")

    plt.grid()
    plt.tight_layout()

    figure_file_name = out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ########### Gen Adv ########################
    file_name = args.model + "_adv_loss"
    plt.figure()
    plt.figure(figsize=(5, 5))

    plt.plot(np.arange(g_gan_loss_val.shape[0])/100, g_gan_loss_val, label='Gen', marker="o")

    plt.plot(np.arange( d_loss_val.shape[0])/100, d_loss_val, label='Disc', marker="+")

    x_label = "Epoch, $10^2$"
    plt.ylabel('Adv Loss')
    plt.xlabel(x_label)

    plt.legend()
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #                  "sr=" + str(int(args.sr)//1000) + " KHz")
    plt.grid()
    plt.tight_layout()
    plt.ylim([0, 4])
    plt.xlim([0, 2])
    plt.xticks([0, 1, 2])

    figure_file_name =  out_dir_name + file_name

    plt.savefig(figure_file_name + ".png", format='png')

def main():

    parser = make_parser()
    args = parser.parse_args()

    if args.sr == '16000' and args.model == "gan_alt_5_multispeaker":

        logs_dir_name = "/multispeaker/sr16000/gan_alt_5/"
        out_dir_name = "../results/learning_curves/multispeaker/sr16000/gan_alt_5/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_autoencoder_gan"

    file_name_logs_val =   "logs" + logs_dir_name + \
                           "multispeaker.r_4.gan.b128.sr_16000_loss_val_autoencoder_gan"

    file_name_logs_train = "logs" + logs_dir_name + \
                           "multispeaker.r_4.gan.b128.sr_16000_loss_train_autoencoder_gan"

    plt_loss_evolution("logs" + logs_fname_evol, out_dir_name)
    plt_loss_epoch(file_name_logs_val, file_name_logs_train, out_dir_name, args)

if __name__ == '__main__':
  main()
