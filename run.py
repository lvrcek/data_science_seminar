# from tqdm import tqdm
from time import time
import numpy as np
import random
import re
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

import joblib

from model import ResNet18
from pileogram import PileogramDataset
import visualizer


NONCHIMERIC_TRAIN = "./2d/nonchimeric"
CHIMERIC_TRAIN = "./2d/chimeric"

EPOCHS = 50
BATCH = 128
PARAM_PATH = 'trained_models/params_res18.pt'


SPECIES = {
    'bs': (0, 3714),
    'cn': (3714, 4342),
    'ec': (4342, 7678),
    'ef': (7678, 11572),
    'lf': (11572, 15184),
    'lm': (15184, 18576),
    'pa': (18576, 22674),
    'sa': (22674, 26288),
    'sc': (26288, 26936),
    'se': (26936, 32618),
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAIN_SPECIES = ['bs', 'cn', 'ec', 'ef', 'lf']
TEST_SPECIES  = ['lm', 'pa', 'sa', 'sc', 'se']
TRAIN_SAMPLES = sum([SPECIES[s][1] - SPECIES[s][0] for s in TRAIN_SPECIES])  # 15184


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Train the ResNet to get the descriptors
def train_nn():
    start_time = time()
    set_seed()
    mode = 'train'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    ds_full = PileogramDataset(NONCHIMERIC_TRAIN, CHIMERIC_TRAIN, transform=transform)
    ds = ds_full[:TRAIN_SAMPLES]
    num_samples = len(ds)
    val_size = round(num_samples * 0.2)
    train_size = num_samples - val_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    # dl_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    net = ResNet18()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use cuda if possible
    # device = torch.device('cpu')  # Force using cpu
    print(f"Using device: {device}")
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.RMSprop(net.parameters(), lr=3e-5)
    history_train = []
    history_val = []
    acc_train = []
    acc_valid = []

    if mode == 'train':
        for epoch in range(EPOCHS):
            total_loss = 0.0
            iteration = 0
            total = 0
            correct = 0
            net.train()

            for data in dl_train:
                iteration += 1
                inputs = data['image'].to(device, non_blocking=True)
                labels = data['label'].to(device, non_blocking=True).float()
                optimizer.zero_grad()
                outputs = net(inputs).squeeze(-1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
                total_loss += loss.item()
                total += labels.size(0)
                predicted = torch.sigmoid(outputs).round()
                # print(predicted)
                correct += (predicted == labels).sum().item()

            # if i % 100 == 99:
            #    print("Epoch: %2d, Step: %5d -> Loss: %.5f" %
            #          (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0
            accuracy = 100*correct/total
            print(f"Epoch {epoch + 1}:\tTrain loss = {total_loss / iteration}\tAccuracy = {round(accuracy, 2)}%")
            history_train.append((epoch + 1, total_loss / iteration))
            acc_train.append((epoch+1, accuracy))

            total_loss = 0.0
            iteration = 0
            total = 0
            correct = 0
            net.eval()

            with torch.no_grad():
                for data in dl_val:
                    iteration += 1
                    images = data['image'].to(device)
                    labels = data['label'].to(device).float()
                    outputs = net(images).squeeze(-1)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    total += labels.size(0)
                    predicted = torch.sigmoid(outputs).round()
                    # print(predicted)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}:\tVal loss = {total_loss / iteration},\tAccuracy = {round(accuracy, 2)}%")
            history_val.append((epoch + 1, total_loss / iteration))
            acc_valid.append((epoch + 1, accuracy))

            if epoch == 0 or acc_valid[-1] > max(acc_valid[:-1]):
                torch.save(net.state_dict(), PARAM_PATH)

        training_time = time()
        print(f"Finished Training. Training time: {training_time - start_time} s")
        visualizer.draw_training_curve(history_train, history_val)
        visualizer.draw_accuracy_curve(acc_train, acc_valid)

    correct = 0
    total = 0
    net.load_state_dict(torch.load(PARAM_PATH))
    net.eval()
    eval_time_start = time()

    # TEST
    ds_test = ds_full[TRAIN_SAMPLES:]
    dl_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for data in dl_test:
            images = data['image'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True).float()
            paths = data['path'][0]
            print(paths)
            print(type(paths))
            outputs = net(images).squeeze(-1)
            total += labels.size(0)
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()

    eval_time_end = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}%.")
    print(f"Evalutaion time: {eval_time_end - eval_time_start} s.")


def extract_descriptors():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds_full = PileogramDataset(NONCHIMERIC_TRAIN, CHIMERIC_TRAIN, transform=transform)
    net = ResNet18()
    net.to(device)
    net.load_state_dict(torch.load(PARAM_PATH))
    net.eval()
    descriptors, targets = [], []
    feature_extractor = torch.nn.Sequential(*list(net.model.children())[:-1])
    for data in ds_full:
        image = data['image'].to(device).unsqueeze(0)
        label = data['label']
        feature = feature_extractor(image).squeeze().squeeze().squeeze().detach().cpu().numpy()
        target = np.array([label])
        descriptors.append(feature)
        targets.append(target)

    descriptors = np.array(descriptors)
    targets = np.array(targets)
    with open('data/descriptors.npy', 'wb') as f1, open('data/targets.npy', 'wb') as f2:
        np.save(f1, descriptors, allow_pickle=True)
        np.save(f2, targets, allow_pickle=True)
    

def train_classifiers():
    descriptors = np.load('data/descriptors.npy', allow_pickle=True)
    targets = np.load('data/targets.npy', allow_pickle=True)

    # Now just train new models on the train dataset, and evaluate on the test dataset
    X_train, X_test = descriptors[:TRAIN_SAMPLES], descriptors[TRAIN_SAMPLES:]
    y_train, y_test = targets[:TRAIN_SAMPLES], targets[TRAIN_SAMPLES:]

    # PCA
    visualizer.do_pca(X_train, y_train, 'pca/train.png')
    visualizer.do_pca(X_test, y_test, 'pca/test.png')
    threads = 16

    svc = False # True
    logreg = False # True
    forest = False # True
    xgboost = False # True

    best_accuracy = 0

    if svc:
        print('SVC:')
        clf = SVC()
        accuracy = train_model(clf, X_train, X_test, y_train, y_test)
        best_model = clf
        best_accuracy = accuracy
        joblib.dump(clf, f'classifiers/svm_clf.joblib')
        print(accuracy)
    else:
        clf = joblib.load('classifiers/svm_clf.joblib')
        accuracy = predict(clf, X_train, X_test, y_train, y_test)
        print(accuracy)



    if logreg:
        print('Logistic regression:')
        clf = LogisticRegression()
        accuracy = train_model(clf, X_train, X_test, y_train, y_test)
        joblib.dump(clf, f'classifiers/logr_clf.joblib')
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
    else:
        clf = joblib.load('classifiers/logr_clf.joblib')
        accuracy = predict(clf, X_train, X_test, y_train, y_test)
        print(accuracy)


    # print('K-nearest neighbors:')
    # clf = KNeighborsClassifier()
    # accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    # joblib.dump(clf, f'classifiers/knn_clf.joblib')
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = clf

    if forest:
        print('Random forest:')
        clf = RandomForestClassifier(n_jobs=threads, n_estimators=800, max_depth=30)
        accuracy = train_model(clf, X_train, X_test, y_train, y_test)
        joblib.dump(clf, f'classifiers/forest_clf.joblib')
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
    else:
        clf = joblib.load('classifiers/forest_clf.joblib')
        accuracy = predict(clf, X_train, X_test, y_train, y_test)
        print(accuracy)


    # print('Decision tree:')
    # clf = DecisionTreeClassifier()
    # accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    # joblib.dump(clf, f'classifiers/tree_clf.joblib')
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = clf

    if xgboost:
        print('XGBoost:')
        clf = XGBClassifier(n_jobs=threads, n_estimators=800, max_depth=30, use_label_encoder=False, eval_metric='logloss')
        accuracy = train_model(clf, X_train, X_test, y_train, y_test)
        joblib.dump(clf, f'classifiers/xgboost_clf.joblib')
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf
    else:
        clf = joblib.load('classifiers/xgboost_clf.joblib')
        accuracy = predict(clf, X_train, X_test, y_train, y_test)
        print(accuracy)

    # print(best_model)
    # print(best_accuracy)


def train_model(model, X_train, X_test, y_train, y_test):
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))
    model_name = re.findall(r'([a-zA-Z]*)\(', str(model))[0] 
    visualizer.plt_confusion_matrix(y_test, y_pred, model_name)
    return metrics.accuracy_score(y_test, y_pred)


def predict(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred, digits=4))
    print(confusion_matrix(y_test, y_pred))
    model_name = re.findall(r'([a-zA-Z]*)\(', str(model))[0]
    visualizer.plt_confusion_matrix(y_test, y_pred, model_name)
    return metrics.accuracy_score(y_test, y_pred)


def main():
    set_seed()
    # train_nn()
    # extract_descriptors()
    train_classifiers()


if __name__ == '__main__':
    main()

