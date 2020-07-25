from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

rcParams['font.family'] = 'monospace'
rcParams['font.size'] = '13'

with open("prob.txt", "r") as f:
    probs = np.array([float(w) for w in f.read().split()])

print(probs[:100])
axis = plt.axes(frame_on=False)

# plt.title("# of occurences of probability values, across all dimensions of vocabulary vectors")

plt.xlabel("probability of feature")
plt.ylabel("# of occurences")
plt.yscale('log')
xs = list(np.arange(0.0, 0.025, 0.005))

plt.axvline(x=1.0 / 200.0, linewidth=3, color="#F57C00")


# plt.yticks(bins, ["10^%s" % b for b in bins])
plt.hist(probs, bins=50, facecolor="#42A5F5")
plt.xticks(xs)

plt.text((1.0 / 200.0) * 1.1, 100, "mean: %s" % (1.0/200.0), color="#FFCC80")
        # color="#212121")

plt.savefig("probabilities.png")
plt.show()

