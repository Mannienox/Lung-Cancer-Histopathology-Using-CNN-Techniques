import matplotlib.pyplot as plt

def plot_loss_curves(epochs, train_loss, test_loss, train_acc, test_acc):
    fig, ax = plt.subplots(1,2, figsize = (12,5))

    ax[0].plot(epochs, train_loss, label = "Train Loss")
    ax[0].plot(epochs, test_loss, label = "Test Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label = "Train Accuracy")
    ax[1].plot(epochs, test_acc, label = "Test Accuracy")
    ax[1].legend()

    fig.suptitle("Loss curves")


def plot_confusion_matrix(actual, pred, labels = None):
    conf_mat = confusion_matrix(actual, pred)
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=labels).plot()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation = 90)
    plt.show()
