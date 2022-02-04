# from tqdm import tqdm
from time import time
import numpy as np
import random
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

import joblib

from model import ResNet18
from pileogram import PileogramDataset
import visualizer

NONCHIMERIC_TRAIN = "./2d/nonchimeric"
CHIMERIC_TRAIN = "./2d/chimeric"

EPOCHS = 50
BATCH = 128
PARAM_PATH = 'trained_models/params_res18.pt'
TRAIN_SAMPLES = 8373


types = {
    0: 'RP',
    1: 'CH',
    2: 'RG',
    3: 'JK',
}


def print_confusion(conf_rep, conf_chim, conf_norm, conf_junk):
    print("%42s" % ('Predicted'))
    print(" " * 21 + "_" * 33)
    print(" " * 20 + "|%10s|%10s|%10s|%10s|" % ('Repeats', 'Chimeric', 'Normal', "Junk"))
    print(" " * 9 + "|" + "%10s" % ('Repeats') + "|%10d|%10d|%10d|%10d|"
          % (conf_rep[0], conf_rep[1], conf_rep[2], conf_rep[3]))
    print("True" + " " * 5 + "|" + "%10s" % ('Chimeric') + "|%10d|%10d|%10d|%10d|"
          % (conf_chim[0], conf_chim[1], conf_chim[2], conf_chim[3]))
    print(" " * 9 + "|" + "%10s" % ('Normal') + "|%10d|%10d|%10d|%10d|"
          % (conf_norm[0], conf_norm[1], conf_norm[2], conf_norm[3]))
    print(" " * 9 + "|" + "%10s" % ('Junk') + "|%10d|%10d|%10d|%10d|"
          % (conf_junk[0], conf_junk[1], conf_junk[2], conf_junk[3]))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Train the ResNet to get the descriptors
def train():
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use cuda if possible
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
    guess_repeat = []
    guess_chim = []
    guess_regular = []
    guess_junk = []
    eval_time_start = time()

    # Don't need this junk
    # with torch.no_grad():
    #     for data in dl_test:
    #         images = data['image'].to(device, non_blocking=True)
    #         labels = data['label'].to(device, non_blocking=True).float()
    #         paths = data['path'][0]
    #         # print(paths)
    #         # print(type(paths))
    #         outputs = net(images).squeeze(-1)
    #         total += labels.size(0)
    #         predicted = torch.sigmoid(outputs).round()
    #         correct += (predicted == labels).sum().item()

    # conf_repeat = (sum([l == 0 for l in guess_repeat]), sum([l == 1 for l in guess_repeat]),
    #                sum([l == 2 for l in guess_repeat]), sum([l == 3 for l in guess_repeat]))
    # conf_chim = (sum([l == 0 for l in guess_chim]), sum([l == 1 for l in guess_chim]),
    #              sum([l == 2 for l in guess_chim]), sum([l == 3 for l in guess_chim]))
    # conf_regular = (sum([l == 0 for l in guess_regular]), sum([l == 1 for l in guess_regular]),
    #                sum([l == 2 for l in guess_regular]), sum([l == 3 for l in guess_regular]))
    # conf_junk = (sum([l == 0 for l in guess_junk]), sum([l == 1 for l in guess_junk]),
    #                sum([l == 2 for l in guess_junk]), sum([l == 3 for l in guess_junk]))

    # print_confusion(conf_repeat, conf_chim, conf_regular, conf_junk)



    # Here I have to test the linear classifier (logistic regression)
    # Which is equivalent to simply evaluating with the last layer of resnet18

    # The rest is the test set
    ds_test = ds_full[TRAIN_SAMPLES:]
    dl_test = DataLoader(ds_test, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for data in dl_test:
            images = data['image'].to(device, non_blocking=True)
            labels = data['label'].to(device, non_blocking=True).float()
            paths = data['path'][0]
            # print(paths)
            # print(type(paths))
            outputs = net(images).squeeze(-1)
            total += labels.size(0)
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()

    eval_time_end = time()
    print(f"Accuracy of the network on the test set: {100 * correct / total}%.")
    print(f"Evalutaion time: {eval_time_end - eval_time_start} s.")

    # Just translate all the descriptors into numpy
    # This is very slow, I should find a better way to do it
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
        # print(image.shape)
        # print(type(label))
        # print(feature.shape)
    descriptors = np.array(descriptors)
    targets = np.array(targets)
    with open('data/descriptors.npy', 'wb') as f1, open('data/targets.npy', 'wb') as f2:
        np.save(f1, descriptors, allow_pickle=True)
        np.save(f2, targets, allow_pickle=True)
    

    # Now just train new models on the train dataset, and evaluate on the test dataset
    X_train, X_test = descriptors[:TRAIN_SAMPLES], descriptors[TRAIN_SAMPLES:]
    y_train, y_test = targets[:TRAIN_SAMPLES], targets[TRAIN_SAMPLES:]

    threads = 1

    print('SVC:')
    clf = SVC()
    accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    best_model = clf
    best_accuracy = accuracy
    joblib.dump(clf, f'classifiers/svm_clf.joblib')

    # print('Logistic regression:')
    # clf = LogisticRegression()
    # accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    # joblib.dump(clf, f'classifiers/logr_clf.joblib')
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = clf

    # print('K-nearest neighbors:')
    # clf = KNeighborsClassifier()
    # accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    # joblib.dump(clf, f'classifiers/knn_clf.joblib')
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = clf

    print('Random forest:')
    clf = RandomForestClassifier(n_jobs=threads, n_estimators=800, max_depth=30)
    accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    joblib.dump(clf, f'classifiers/forest_clf.joblib')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

    # print('Decision tree:')
    # clf = DecisionTreeClassifier()
    # accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    # joblib.dump(clf, f'classifiers/tree_clf.joblib')
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     best_model = clf

    print('XGBoost:')
    clf = XGBClassifier(n_jobs=threads, n_estimators=800, max_depth=50, use_label_encoder=False, eval_metric='logloss')
    accuracy = train_model(clf, X_train, X_test, y_train, y_test)
    joblib.dump(clf, f'classifiers/xgboost_clf.joblib')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf

    print(best_model)
    print(best_accuracy)


def train_model(model, X_train, X_test, y_train, y_test):
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    train()
    # extract()
