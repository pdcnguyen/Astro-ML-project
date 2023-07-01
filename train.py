import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score
import albumentations as A
import numpy as np
from copy import deepcopy
from torch.utils.data import WeightedRandomSampler

from dataset import SDSSData, SDSSData_train, SDSSData_val, SDSSData_test
from model import CNN_with_Unet

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")


class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.state_dict = None
        self.best_f1 = 0

    def early_stop(self, validation_loss, model, f1):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.state_dict = deepcopy(model.state_dict())
            self.best_f1 = f1
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif (
            validation_loss < (self.min_validation_loss + self.min_delta) and validation_loss > self.min_validation_loss
        ):
            self.counter += self.patience / 3.0
            if self.counter >= self.patience:
                return True
        return False

    def load_best_model(self):
        return self.state_dict

    def get_best_f1(self):
        return self.best_f1


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
        running_f1 += f1_score(labels.cpu(), torch.argmax(outputs.cpu(), dim=1), zero_division=0)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 100 - 1:  # print every 100 batches
            avg_loss = running_loss / 100
            avg_acc = (running_accuracy / 100) * 100
            avg_f1 = running_f1 / 100
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


def prepare_training(params, model, transform, is_tunning):
    data = SDSSData(params["dist_from_center"], is_tunning=is_tunning)

    trainset = SDSSData_train(data, transform=transform)
    sampler = WeightedRandomSampler(weights=trainset.sample_weights, num_samples=len(trainset), replacement=True)
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=params["batch_size"], num_workers=2)

    valset = SDSSData_val(data, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params["learning_rate"])

    early_stopper = EarlyStopper(patience=2, min_delta=0.05)

    return trainloader, valloader, criterion, optimizer, early_stopper


def train_and_evaluate(params, model, transform, trial):
    trainloader, valloader, criterion, optimizer, early_stopper = prepare_training(
        params, model, transform, is_tunning=True
    )

    for epoch_index in range(10):
        train_one_epoch(model, optimizer, criterion, trainloader)

        val_loss, val_accuracy, val_f1 = validate(model, criterion, valloader)

        if early_stopper.early_stop(val_loss, model, val_f1):
            break

        trial.report(val_f1, epoch_index)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return early_stopper.get_best_f1()


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

    trainloader, valloader, criterion, optimizer, early_stopper = prepare_training(
        params, model, transform, is_tunning=False
    )

    testset = SDSSData_test(params["dist_from_center"])
    testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    for epoch_index in range(10):
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
    print(f"Using img_80: Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.1f}%, Test f1: {test_f1:.2f}")
    print("***************************************************")

    return model


def objective(trial, transform):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "dist_from_center": trial.suggest_categorical("dist_from_center", [10, 15, 20]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
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

    f1 = train_and_evaluate(params, model, transform, trial)

    return f1


def tune_parameters(n_trials, study_name, transform=None):
    print("Tuning in process...")
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
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
