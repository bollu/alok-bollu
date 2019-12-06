from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams['font.family'] = 'monospace'
rcParams['font.size'] = '13'


with open("entropy.txt", "r") as f:
    xs = np.array([float(x) for x in f.read().split()])
axis = plt.axes(frame_on=False)

plt.hist(xs, 100, facecolor="#42A5F5")
# cats
plt.text(6.2739 + 0.0005, 4200, "cats", color="#212121")
plt.axvline(x=6.2739, linewidth=3, color="#F57C00")

xs = list(np.arange(6.24, 6.30, 0.01))

# the
plt.axvline(x=6.2918, linewidth=3, color="#F57C00")
plt.text(6.2918 + 0.0005, 4200, "the", color="#212121")


# the
plt.axvline(x=6.26, linewidth=3, color="#F57C00")
plt.text(6.26 + 0.0005, 4200, "giga", color="#212121")

plt.xlabel("entropy")
plt.ylabel("# of occurences")


plt.savefig("entropy.png")
plt.show()

