from matplotlib import pyplot as plt


def draw_training_curve(history_train, history_test):
    plt.figure()
    epochs = [h[0] for h in history_train]
    loss_train = [h[1] for h in history_train]
    loss_test = [h[1] for h in history_test]
    plt.plot(epochs, loss_train, label='train')
    plt.plot(epochs, loss_test, label='validation')
    plt.title("Model loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')
#    plt.show()


def draw_accuracy_curve(acc_train, acc_valid):
    plt.figure()
    epochs = [h[0] for h in acc_train]
    acc_train = [h[1] for h in acc_train]
    acc_valid = [h[1] for h in acc_valid]
    plt.plot(epochs, acc_train, label='train')
    plt.plot(epochs, acc_valid, label='validation')
    plt.title("Model accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('accuracy.png')
#    plt.show()