import numpy as np
import matplotlib.pyplot as plt

# the logic unit of everything:
# the AND NOT gate
# A NAND B = ~(A&B)

def main():
    x = np.arange(12)
    y = np.square(x)
    plt.plot(x, y)
    plt.show()

    x_bins = ['{0:04b}'.format(i) for i in x]
    y_bins = ["{0:07b}".format(i) for i in y]

    print(x_bins)
    print(y_bins)


main()
