import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from torch.utils.data import WeightedRandomSampler

from dataset import SDSSData, SDSSData_train, SDSSData_val, SDSSData_test
from model import CNN_with_Unet
from early_stopper import EarlyStopper
from config import DIST_FROM_CENTER, MAX_EPOCH

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")


def train_one_epoch(model, optimizer, criterion, trainloader):
    model.train(True)

    running_loss = 0.0
    running_accuracy = 0.0
    running_metric_score = 0.0

    epoch_run_results = []

    for batch_index, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # shape: [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_accuracy += correct / len(labels)

        loss = criterion(outputs, labels)
        running_metric_score += f1_score(
            labels.cpu(), torch.argmax(outputs.cpu(), dim=1), zero_division=0
        )
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 100 - 1:  # print every 100 batches
            avg_loss = running_loss / 100
            avg_acc = (running_accuracy / 100) * 100
            avg_metric_score = running_metric_score / 100
            epoch_run_results.append((avg_loss, avg_acc, avg_metric_score))
            running_loss = 0.0
            running_accuracy = 0.0
            running_metric_score = 0.0

    return epoch_run_results


def validate(model, criterion, valloader, verbose=False):
    model.train(False)
    running_loss = 0.0
    running_accuracy = 0.0
    running_metric_score = 0.0

    for i, data in enumerate(valloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / len(labels)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_metric_score += f1_score(
                labels.cpu(), torch.argmax(outputs.cpu(), dim=1), zero_division=0
            )

    avg_loss = running_loss / len(valloader)
    avg_acc = (running_accuracy / len(valloader)) * 100
    avg_metric_score = running_metric_score / len(valloader)

    if verbose:
        print(
            f"Val Loss: {avg_loss:.3f}, Val Accuracy: {avg_acc:.1f}%, Val metric_score: {avg_metric_score:.2f}"
        )
        print("***************************************************")

    return avg_loss, avg_acc, avg_metric_score


def prepare_training(params, model, trainset, valset):
    sampler = WeightedRandomSampler(
        weights=trainset.sample_weights, num_samples=len(trainset), replacement=True
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, sampler=sampler, batch_size=params["batch_size"], num_workers=2
    )

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=params["batch_size"], num_workers=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(
        model.parameters(), lr=params["learning_rate"]
    )

    early_stopper = EarlyStopper(patience=3, min_delta=0.03)

    return trainloader, valloader, criterion, optimizer, early_stopper


def train_and_evaluate(params, model, trainset, valset, trial):
    trainloader, valloader, criterion, optimizer, early_stopper = prepare_training(
        params, model, trainset, valset
    )

    for epoch_index in range(MAX_EPOCH):
        train_one_epoch(model, optimizer, criterion, trainloader)

        val_loss, val_accuracy, score = validate(model, criterion, valloader)

        if early_stopper.early_stop(model, val_loss, score):
            break

        trial.report(score, epoch_index)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return early_stopper.get_best_metric_score()


def objective(trial, transform, trainset, valset):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "drop_out": trial.suggest_float("drop_out", 0.1, 0.3),
        "hidden_nodes": trial.suggest_categorical("hidden_nodes", [256, 512, 1024]),
    }

    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=DIST_FROM_CENTER,
        drop_out=params["drop_out"],
        hidden_nodes=params["hidden_nodes"],
    )
    model = model.to(device)

    metric_score = train_and_evaluate(params, model, trainset, valset, trial)

    return metric_score


def tune_parameters(n_trials, study_name, transform=None):
    print("Tuning in process...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=296),
        storage="sqlite:///results.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    data = SDSSData(DIST_FROM_CENTER, is_tunning=True)
    trainset = SDSSData_train(data, transform=transform)
    valset = SDSSData_val(data, transform=transform)

    func = lambda trial: objective(trial, transform, trainset, valset)
    study.optimize(func, n_trials=n_trials, n_jobs=1, show_progress_bar=True)


def full_train_and_test(params, transform=None):
    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=DIST_FROM_CENTER,
        drop_out=params["drop_out"],
        hidden_nodes=params["hidden_nodes"],
    )
    model = model.to(device)

    data = SDSSData(DIST_FROM_CENTER, is_tunning=False)
    trainset = SDSSData_train(data, transform=transform)

    valset = SDSSData_val(data, transform=transform)

    trainloader, valloader, criterion, optimizer, early_stopper = prepare_training(
        params, model, trainset, valset
    )

    for epoch_index in range(MAX_EPOCH):
        print(f"Epoch: {epoch_index + 1}")

        epoch_run_results = train_one_epoch(model, optimizer, criterion, trainloader)

        for loss, accuracy, metric_score in epoch_run_results:
            print(
                f"loss: {loss:.3f}, accuracy: {accuracy:.1f}%, metric_score: {metric_score:.2f}"
            )
        print("VAL SET")
        val_loss, val_accuracy, val_metric_score = validate(
            model, criterion, valloader, verbose=True
        )

        if early_stopper.early_stop(model, val_loss, val_metric_score):
            model.load_state_dict(early_stopper.load_best_model())
            break

    # test
    testset = SDSSData_test(DIST_FROM_CENTER)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=True, num_workers=2
    )

    model.train(False)
    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()

            accuracy = correct / len(labels) * 100

            loss = criterion(outputs, labels)
            loss = loss.item()

            metric_score = f1_score(
                labels.cpu(), torch.argmax(outputs.cpu(), dim=1), zero_division=0
            )

            auc_score = roc_auc_score(labels.cpu(), torch.argmax(outputs.cpu(), dim=1))
            print("TEST SET")
            print(
                f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.1f}%, Test f1_score: {metric_score:.2f}, Test auc_score: {auc_score:.2f}\n"
            )
            print(
                classification_report(
                    labels.cpu(),
                    torch.argmax(outputs.cpu(), dim=1),
                    target_names=["galaxies", "stars"],
                )
            )
            print("***************************************************")

    return model
