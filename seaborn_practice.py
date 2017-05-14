import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)


sns.set()  # reset all default parameters
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# sns.set_palette("hls", 8)
# sns.set_palette("husl")
# sns.set_palette("PuBuGn_d")

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))

# sinplot()

gammas = sns.load_dataset("gammas")
ax = sns.tsplot(data=gammas,
                time="timepoint",
                value="BOLD signal",
                unit="subject",
                condition="ROI",
                ci=[68, 95])

# num_points = 1000
# x = np.linspace(0, 15, num_points)
# data = np.sin(x) + np.random.rand(10, num_points) + np.random.randn(10, 1)
# ax = sns.tsplot(data=data,
#                 ci=[68, 95, 99.7])

plt.show()