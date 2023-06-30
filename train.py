import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score
import albumentations as A
import numpy as np
from copy import deepcopy

from dataset import SDSSData, SDSSData_train, SDSSData_test
from model import CNN_with_Unet

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.state_dict = None
        self.best_accuracy = 0

    def early_stop(self, validation_loss, model, accuracy):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.state_dict = deepcopy(model.state_dict())
            self.best_accuracy = accuracy
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif (
            validation_loss < (self.min_validation_loss + self.min_delta) and validation_loss > self.min_validation_loss
        ):
            self.counter += self.patience / 3.0
            if self.counter >= self.patience + 1:
                return True
        return False

    def load_best_model(self):
        return self.state_dict

    def get_best_accuracy(self):
        return self.best_accuracy


def train_one_epoch(model, optimizer, criterion, trainloader):
    model.train(True)

    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0

    epoch_run_results = []

    for batch_index, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # shape: [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_accuracy += correct / len(labels)

        loss = criterion(outputs, labels)
        running_f1 += f1_score(labels.cpu(), torch.argmax(outputs.cpu(), dim=1))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 20 == 20 - 1:  # print every 20 batches
            avg_loss = running_loss / 20
            avg_acc = (running_accuracy / 20) * 100
            avg_f1 = running_f1 / 20
            epoch_run_results.append((avg_loss, avg_acc, avg_f1))
            running_loss = 0.0
            running_accuracy = 0.0
            running_f1 = 0.0

    return epoch_run_results


def validate(model, criterion, valloader):
    model.train(False)
    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0

    for i, data in enumerate(valloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / len(labels)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_f1 += f1_score(labels.cpu(), torch.argmax(outputs.cpu(), dim=1), zero_division=0)

    avg_loss = running_loss / len(valloader)
    avg_acc = (running_accuracy / len(valloader)) * 100
    avg_f1 = running_f1 / len(valloader)

    return avg_loss, avg_acc, avg_f1


def train_and_evaluate(params, model, transform, trial):
    data = SDSSData(params["dist_from_center"], is_tunning=True)
    trainset = SDSSData_train(data, transform=transform)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 5, len(trainset) // 5])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params["learning_rate"])

    early_stopper = EarlyStopper(patience=2, min_delta=0.05)
    for epoch_index in range(15):
        train_one_epoch(model, optimizer, criterion, trainloader)

        val_loss, val_accuracy, val_f1 = validate(model, criterion, valloader)

        if early_stopper.early_stop(val_loss, model, val_accuracy):
            break

        trial.report(val_accuracy, epoch_index)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return early_stopper.get_best_accuracy()


def hard_train_and_test(params, transform=None):
    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=params["dist_from_center"],
        drop_out=params["drop_out"],
        hidden_nodes=params["hidden_nodes"],
    )
    model = model.to(device)

    data = SDSSData(params["dist_from_center"])
    trainset = SDSSData_train(data, transform=transform)
    testset = SDSSData_test(data)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 5, len(trainset) // 5])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params["learning_rate"])

    early_stopper = EarlyStopper(patience=2, min_delta=0.05)

    for epoch_index in range(15):
        print(f"Epoch: {epoch_index + 1}\n")

        epoch_run_results = train_one_epoch(model, optimizer, criterion, trainloader)

        for loss, accuracy, f1 in epoch_run_results:
            print(f"loss: {loss:.3f}, accuracy: {accuracy:.1f}%, f1: {f1:.2f}")

        val_loss, val_accuracy, val_f1 = validate(model, criterion, valloader)
        print(f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.1f}%, Val f1: {val_f1:.2f}")
        print("***************************************************")

        if early_stopper.early_stop(val_loss, model, val_accuracy):
            model.load_state_dict(early_stopper.load_best_model())
            break

    # test
    test_loss, test_accuracy, test_f1 = validate(model, criterion, testloader)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.1f}%, Test f1: {test_f1:.2f}")
    print("***************************************************")

    return model


def predict(model, data):
    testloader = torch.utils.data.DataLoader(data, batch_size=70, shuffle=False, num_workers=2)

    prediction = []

    for i, data in enumerate(testloader):
        inputs = data.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            prediction.append(torch.argmax(outputs, dim=1))

    return torch.hstack(prediction)


def objective(trial, transform):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "dist_from_center": trial.suggest_categorical("dist_from_center", [10, 15, 20]),
        "batch_size": trial.suggest_categorical("batch_size", [50, 70, 100]),
        "drop_out": trial.suggest_float("drop_out", 0.1, 0.3),
        "hidden_nodes": trial.suggest_categorical("hidden_nodes", [256, 512, 1024]),
    }

    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=params["dist_from_center"],
        drop_out=params["drop_out"],
        hidden_nodes=params["hidden_nodes"],
    )
    model = model.to(device)

    accuracy = train_and_evaluate(params, model, transform, trial)

    return accuracy


def tune_parameters(n_trials, study_name, transform=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=296),
        storage="sqlite:///results.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    func = lambda trial: objective(trial, transform)
    study.optimize(
        func, n_trials=n_trials, n_jobs=1
    )  # more n_jobs if you want parallelization, might be buggy combining with seed


if __name__ == "__main__":
    params = {
        "batch_size": 70,
        "dist_from_center": 15,
        "drop_out": 0.20879658201895385,
        "hidden_nodes": 256,
        "learning_rate": 0.0005881835636133336,
        "optimizer": "Adam",
    }
    hard_train_and_test(params)
