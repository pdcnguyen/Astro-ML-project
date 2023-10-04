import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import WeightedRandomSampler

from dataset import SDSSData, SDSSData_train, SDSSData_val, SDSSData_test
from model import CNN_with_Unet
from earlyStopper import EarlyStopper

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")
DIST_FROM_CENTER = 15
MAX_EPOCH = 20


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
        running_metric_score += roc_auc_score(
            labels.cpu(), torch.argmax(outputs.cpu(), dim=1)
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


def validate(model, criterion, valloader, is_tunning=True):
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
            running_metric_score += roc_auc_score(
                labels.cpu(), torch.argmax(outputs.cpu(), dim=1)
            )
            if not is_tunning:
                print(
                    classification_report(
                        labels.cpu(),
                        torch.argmax(outputs.cpu(), dim=1),
                        target_names=["gals", "stars"],
                    )
                )

    avg_loss = running_loss / len(valloader)
    avg_acc = (running_accuracy / len(valloader)) * 100
    avg_metric_score = running_metric_score / len(valloader)

    return avg_loss, avg_acc, avg_metric_score


def prepare_training(params, model, trainset, valset):
    sampler = WeightedRandomSampler(
        weights=trainset.sample_weights, num_samples=len(trainset), replacement=True
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, sampler=sampler, batch_size=params["batch_size"], num_workers=2
    )

    valloader = torch.utils.data.DataLoader(
        valset, batch_size=params["batch_size"], shuffle=False, num_workers=2
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(
        model.parameters(), lr=params["learning_rate"]
    )

    early_stopper = EarlyStopper(patience=3, min_delta=0.05)

    return trainloader, valloader, criterion, optimizer, early_stopper


def train_and_evaluate(params, model, trainset, valset, trial):
    trainloader, valloader, criterion, optimizer, early_stopper = prepare_training(
        params, model, trainset, valset
    )

    for epoch_index in range(MAX_EPOCH):
        train_one_epoch(model, optimizer, criterion, trainloader)

        val_loss, val_accuracy, score = validate(model, criterion, valloader)

        if early_stopper.early_stop(val_loss, model, score):
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

    metric_score = train_and_evaluate(params, model, transform, trainset, valset, trial)

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
    trainset = WeightedRandomSampler(
        weights=trainset.sample_weights, num_samples=len(trainset), replacement=True
    )
    valset = SDSSData_val(data, transform=transform)

    func = lambda trial: objective(trial, transform, trainset, valset)
    study.optimize(func, n_trials=n_trials, n_jobs=1, show_progress_bar=True)


def hard_train_and_test(params, transform=None):
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

    testset = SDSSData_test(DIST_FROM_CENTER)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=params["batch_size"], shuffle=True, num_workers=2
    )

    for epoch_index in range(MAX_EPOCH):
        print(f"Epoch: {epoch_index + 1}")

        epoch_run_results = train_one_epoch(model, optimizer, criterion, trainloader)

        for loss, accuracy, metric_score in epoch_run_results:
            print(
                f"loss: {loss:.3f}, accuracy: {accuracy:.1f}%, metric_score: {metric_score:.2f}"
            )

        val_loss, val_accuracy, val_metric_score = validate(model, criterion, valloader)
        print(
            f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.1f}%, Val metric_score: {val_metric_score:.2f}"
        )
        print("***************************************************")

        if early_stopper.early_stop(val_loss, model, val_accuracy):
            model.load_state_dict(early_stopper.load_best_model())
            break

    # test
    test_loss, test_accuracy, test_metric_score = validate(
        model, criterion, testloader, is_tunning=False
    )
    print(
        f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.1f}%, Test metric_score: {test_metric_score:.2f}"
    )
    print("***************************************************")

    return model
