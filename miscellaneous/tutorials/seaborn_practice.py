import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

def load_run(filename):
	df = pd.read_csv(filename)
	print(df[['val_loss']])
	plt.plot(df[['epoch']], df[['val_loss']])
	# plt.plot(df[['epoch']], df[['val_kl_loss']])
	# plt.plot(df[['epoch']], df[['val_reconstruction_loss']])


sns.set()  # reset all default parameters
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# sns.set_palette("hls", 8)
# sns.set_palette("husl")
# sns.set_palette("PuBuGn_d")

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))

# sinplot()

# experiment = 'experiment_optimal_network_convolutional_latent_pong'
# run_name = 'cvae_atari_entangled_pong_with_latent_image_14_May_23_39_29_batch_size_1_beta_1.0_epochs_10_filters_32_kernel_size_6_latent_filters_1_loss_vae_loss_optimizer_adam'
# filename = './summaries/' + experiment + '/' + run_name + '/log.csv'
# load_run(filename)

gammas = sns.load_dataset("gammas")
ax = sns.tsplot(data=gammas,
                time="timepoint",
                value="BOLD signal",
                unit="subject",
                condition="ROI",
                ci=[95])

# num_points = 1000
# x = np.linspace(0, 15, num_points)
# data = np.sin(x) + np.random.rand(10, num_points) + np.random.randn(10, 1)
# ax = sns.tsplot(data=data,
#                 ci=[68, 95, 99.7])


plt.show()