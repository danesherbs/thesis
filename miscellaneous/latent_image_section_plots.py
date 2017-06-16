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
					shade_lowest=True,
					n_levels=60,
					cmap='plasma')
	# axis_limit = 120
	# ax.set(xlim=(-axis_limit, axis_limit))
	# ax.set(ylim=(-axis_limit, axis_limit))

# make parbo
def multivariate_parabola(X, Y, x_mu, y_mu):
	return - (X - x_mu)**2 - (Y - y_mu)**2


# main function
def main():
	limit = 5
	x = np.linspace(-limit, limit, 1000)
	y = np.linspace(-limit, limit, 1000)
	X, Y = np.meshgrid(x, y)
	Z1 = multivariate_parabola(X, Y, -2.5, -2.5)
	Z2 = multivariate_parabola(X, Y, 2.5, 2.5)
	Z = Z1 + Z2

	levels = 100
	plt.contourf(X, Y, Z, levels, cmap='plasma')
	plt.show()

	# plot_contours(x, y)
	# # plt.savefig(base_name + 'colour_0' + extension)
	# plt.show()

# run main function
if __name__ == '__main__':
	main()