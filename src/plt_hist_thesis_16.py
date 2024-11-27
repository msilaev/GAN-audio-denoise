# parser
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
# -------------------
def make_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model')
    parser.add_argument('--speaker')
    parser.add_argument('--sr')

    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()

    if (args.sr == '16000' and args.speaker == "multi"):

        fig_save_dir = "../results/MOS/"

        file_name_gan = "../results/MOS/scores_16_gan.csv"
        file_name_gan_5 = "../results/MOS/scores_16_gan_alt_5.csv"
        file_name_gan_3 = "../results/MOS/scores_16_gan_alt_3.csv"
        file_name_adiounet = "../results/MOS/scores_16_audiounet.csv"
        #file_name_sg = "logs" + sg_logs_dir_name + "scores_sg.csv"

        plot_histogram( fig_save_dir,  args.sr, file_name_gan_5 )
        #plot_skatter(file_name_gan, fig_save_dir, args.sr, file_name_adiounet)


def plot_skatter(file_name, fig_save_dir, sr, file_name_adiounet ):

    font_size = 20

    if sr == "16000":
        title_str = "4->16 KHz"

    elif sr == "48000":
        title_str = "16->48 KHz"

    df = pd.read_csv(file_name)
     # Assuming MOS_hr and MOS_pr are pandas Series from your dataframe 'df'
    MOS_hr = df["P808_MOS hr"]
    MOS_pr = df["P808_MOS pr"]

    df_aunet = pd.read_csv(file_name_adiounet)
    MOS_pr_aunet = df_aunet["P808_MOS pr"]

    # Create the scatter plot
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(5, 5))

    #plt.figure(figsize=(8, 6))  # Adjust the size as needed
    ax_1.scatter(MOS_hr, MOS_pr, color='blue', alpha=0.7, edgecolors='k')
    # Add labels and title

    # Labels
    ax_1.set_xlabel("MOS hr", fontsize=font_size)
    ax_1.set_ylabel("MOS pr", fontsize=font_size)
    ax_1.tick_params(axis='both', which='major', labelsize=font_size)

    ax_1.set_xlim([0, 5])
    ax_1.set_ylim([0, 5])
    ax_1.set_xticks([0, 1,2,3,4, 5])
    ax_1.grid()
    fig_1.tight_layout()  # Automatically adjusts subplot parameters
    fig_1.savefig(fig_save_dir + "ScatterMOS.png", format='png')


def plot_histogram( fig_save_dir, sr,  file_name_gan_5 ):

    font_size = 20

    if sr == "16000":
        title_str = "4->16 KHz"

    #elif sr == "48000":
    #    title_str = "16->48 KHz"

    df_gan_5 = pd.read_csv(file_name_gan_5)

    SNR_pr_gan_5 = df_gan_5["snr_pr"]
    SNR_noisy = df_gan_5["snr_noisy"]
    MOS_pr_gan_5 = df_gan_5["P808_MOS pr"]

    MOS_hr = df_gan_5["P808_MOS hr"]
    MOS_pr = df_gan_5["P808_MOS pr"]
    MOS_noisy = df_gan_5["P808_MOS noisy"]

    BAK_hr = df_gan_5["BAK hr"]
    BAK_pr = df_gan_5["BAK pr"]

    SIG_hr = df_gan_5["SIG hr"]
    SIG_pr = df_gan_5["SIG pr"]

    OVRL_hr = df_gan_5["OVRL hr"]
    OVRL_pr = df_gan_5["OVRL pr"]

    fig_0, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
    fig_0.subplots_adjust(hspace=0.3)  # Adjust spacing between subplots

    # Subplot 1
    fixed_bins = np.linspace(0, 5, 51)  # 30 bins from 2 to 5
    ax_0 = ax[0]
    ax_0.hist(MOS_hr, bins=fixed_bins, alpha=1.0, color="blue", label="clean")
    ax_0.hist(MOS_noisy, bins=fixed_bins, alpha=0.5, color="red", label="noisy")
    ax_0.set_xlim([2, 5])
    ax_0.set_ylim([0, 500])

    ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.set_xlabel("MOS", fontsize=font_size)

    ax_0.tick_params(axis='both', which='major', labelsize=font_size)
    ax_0.legend(fontsize=font_size)
    ax_0.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_0.set_title("Audiounet vs. HR vs. GAN", fontsize=font_size)

    ax_0 = ax[1]
    ax_0.hist(MOS_hr, bins=fixed_bins, alpha=1.0, color="blue", label="clean")
#    ax_0.hist(MOS_noisy, bins=fixed_bins, alpha=0.5, color="red", label="noisy")
    ax_0.hist(MOS_pr, bins=fixed_bins, alpha=0.5, color="black", label="denoised")
    ax_0.set_xlim([2, 5])
    ax_0.set_ylim([0, 500])


    ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.set_xlabel("MOS", fontsize=font_size)

    #ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.tick_params(axis='both', which='major', labelsize=font_size)
    ax_0.legend(fontsize=font_size)
    ax_0.grid(color='gray', linestyle='--', linewidth=0.5)

    ax_0 = ax[2]
    ax_0.hist(MOS_noisy, bins=fixed_bins, alpha=1.0, color="red", label="noisy")
    #    ax_0.hist(MOS_noisy, bins=fixed_bins, alpha=0.5, color="red", label="noisy")
    ax_0.hist(MOS_pr, bins=fixed_bins, alpha=0.5, color="black", label="denoised")
    ax_0.set_xlim([2, 5])
    ax_0.set_ylim([0, 500])


    ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.set_xlabel("MOS", fontsize=font_size)

    # ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.tick_params(axis='both', which='major', labelsize=font_size)
    ax_0.legend(fontsize=font_size)
    ax_0.grid(color='gray', linestyle='--', linewidth=0.5)

    # Final layout adjustment
    plt.tight_layout()
    plt.show()
    fig_0.savefig(fig_save_dir + "MOS.png", format='png')

if __name__ == "__main__":
    main()