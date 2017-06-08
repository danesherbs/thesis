import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# set seaborn theme
sns.set()  # reset all default parameters
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))

plt.rcParams.update({
                    'font.family' : 'sans-serif',
                    'font.sans-serif' : 'Arial',
                    'font.style' : 'normal',
                    'xtick.labelsize' : 10,
                    'ytick.labelsize' : 10,
                    'axes.labelsize' : 15,
                    'mathtext.fontset' : 'stixsans',
                    'mathtext.default': 'regular',
                    'text.usetex' : True,
                    'text.latex.unicode' : True
                    })


# plot contours
def plot_contours(x, y):
	ax = sns.kdeplot(x, y,
					shade=True,
					shade_lowest=False,
					n_levels=10,
					cmap='Reds')
	axis_limit = 5
	ax.set(xlim=(-axis_limit, axis_limit))
	ax.set(ylim=(-axis_limit, axis_limit))

# plot scatter
def plot_scatter(x, y, colour):
	df = pd.DataFrame()
	df['x'] = x
	df['y'] = y
	df['colour'] = colour
	ax = sns.lmplot('x', 'y',
		           data=df,
		           hue='colour',
		           palette=sns.xkcd_palette([colour]),
		           legend=False,
		           fit_reg=False)
	axis_limit = 3
	ax.set(xlim=(-axis_limit, axis_limit))
	ax.set(ylim=(-axis_limit, axis_limit))
	ax.set(xlabel=r"$z_1$")
	ax.set(ylabel=r"$z_2$")

# main function
def main():
	base_name = 'variational_autoencoder_latent_space_'
	extension = '.eps'

	mean, cov = [-1, 1], [(0.3, 0), (0, 0.3)]
	x, y = np.random.multivariate_normal(mean, cov, size=300).T
	plot_scatter(x, y, colors[0])
	plt.savefig(base_name + 'colour_0' + extension)

	mean, cov = [-1, -1.5], [(0.5, 0.1), (0.3, 0.4)]
	x, y = np.random.multivariate_normal(mean, cov, size=300).T
	plot_scatter(x, y, colors[1])
	plt.savefig(base_name + 'colour_1' + extension)

	mean, cov = [0.7, 0], [(0.3, -0.2), (0.2, 0.7)]
	x, y = np.random.multivariate_normal(mean, cov, size=300).T
	plot_scatter(x, y, colors[3])
	plt.savefig(base_name + 'colour_2' + extension)

	mean, cov = [1, 1], [(0.1, 0), (0.1, 0.1)]
	x, y = np.random.multivariate_normal(mean, cov, size=300).T
	plot_scatter(x, y, colors[4])
	plt.savefig(base_name + 'colour_3' + extension)

# run main function
if __name__ == '__main__':
	main()