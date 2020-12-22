import matplotlib.pyplot as plt

def plot_losses(log_data, filename):
    (loss_line,) = plt.plot("iter", "loss", data=log_data, label="loss")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.legend(handles=[loss_line])

    plt.savefig(filename)
