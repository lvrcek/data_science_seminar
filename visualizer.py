from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    # plt.show()

def do_pca(X, y, path):
    pca = PCA(num_components=2)
    # descriptors = np.load(f'{path}/descriptors.npy', allow_pickle=True)
    # labels = np.load(f'{path}/targets.npy', allow_pickle=True)
    X_0 = X[(y==0).squeeze()]
    X_1 = X[(y==1).squeeze()]
    plt.figure()
    plt.scatter(X_0[:, 0], X_0[:, 1], color='#F8776D', marker='o', s=10, edgecolors='black', linewidths=0.5)
    plt.scatter(X_1[:, 0], X_1[:, 1], color='#01BFC4', marker='o', s=10, edgecolors='black', linewidths=0.5)
    plt.grid()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(path)
    

def plt_confusion_matrix(y_test, y_pred, title):
    # https://stackoverflow.com/questions/20927368/how-to-normalize-a-confusion-matrix

    cm = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    f, axes = plt.subplots(1, 2,figsize=(9,3))
    
    if len(title) != 0:
        f.suptitle(title,fontsize=20,y=1.05)
        
    sns.heatmap(cm, annot=True, fmt='d',cmap=plt.cm.Reds,ax=axes[0])
    
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cmn, annot=True, fmt='.2f',cmap=plt.cm.Blues,ax=axes[1])
    
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
        
    # plt.show(block=False)
    plt.savefig(f'matrix/{title}.png')
