import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from dataset import SDSSData, SDSSData_train, SDSSData_test
from model import CNN_with_Unet

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")


def train_one_epoch(model, optimizer, criterion, trainloader):
    model.train(True)

    running_loss = 0.0
    running_accuracy = 0.0

    for batch_index, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # shape: [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        # running_accuracy += correct / batch_size

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # if batch_index % print_batch == print_batch - 1:  # print every 100 batches
        #     avg_loss_across_batches = running_loss / print_batch
        #     avg_acc_across_batches = (running_accuracy / print_batch) * 100
        #     print(
        #         "Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%".format(
        #             batch_index + 1, avg_loss_across_batches, avg_acc_across_batches
        #         )
        #     )
        #     running_loss = 0.0
        #     running_accuracy = 0.0


def build_model(params):
    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=params["dist_from_center"],
        drop_out=params["drop_out"],
    )
    model = model.to(device)

    return model


def train_and_evaluate(params, model, trial):
    data = SDSSData(1, params["dist_from_center"])
    trainset = SDSSData_train(data)
    # testset = SDSSData_test(data)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params["learning_rate"])

    for epoch_index in range(params["num_epochs"]):
        print(f"Epoch: {epoch_index + 1}\n")

        train_one_epoch(model, optimizer, criterion, trainloader)

        model.train(False)
        running_loss = 0.0
        running_accuracy = 0.0

        for i, data in enumerate(valloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)  # shape: [batch_size, 10]
                correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                running_accuracy += correct / params["batch_size"]
                loss = criterion(outputs, labels)  # One number, the average batch loss
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(valloader)
        avg_acc_across_batches = (running_accuracy / len(valloader)) * 100

        trial.report(avg_acc_across_batches, epoch_index)

        print("Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%".format(avg_loss_across_batches, avg_acc_across_batches))
        print("***************************************************")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_acc_across_batches


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "dist_from_center": trial.suggest_categorical("dist_from_center", [10, 15, 20]),
        "batch_size": trial.suggest_categorical("batch_size", [20, 50, 100]),
        "drop_out": trial.suggest_float("drop_out", 0.1, 0.3),
        "num_epochs": trial.suggest_int("num_epochs", 5, 15),
    }

    model = build_model(params)
    accuracy = train_and_evaluate(params, model, trial)

    return accuracy


def tuning():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        storage="sqlite:///example.db",
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))


def hard_train(learning_rate, optimizer, dist_from_center, batch_size, drop_out, num_epochs):
    model = CNN_with_Unet(
        in_channels=5,
        out_channels=1,
        num_of_class=len(classes),
        dist_from_center=dist_from_center,
        drop_out=drop_out,
    )
    model = model.to(device)

    data = SDSSData(1, dist_from_center)
    trainset = SDSSData_train(data)
    testset = SDSSData_test(data)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    for epoch_index in range(num_epochs):
        print(f"Epoch: {epoch_index + 1}\n")

        train_one_epoch(model, optimizer, criterion, trainloader)

        model.train(False)
        running_loss = 0.0
        running_accuracy = 0.0

        for i, data in enumerate(valloader):
            inputs, labels = data[0].to(device), data[1].to(device)

            with torch.no_grad():
                outputs = model(inputs)  # shape: [batch_size, 10]
                correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
                running_accuracy += correct / batch_size
                loss = criterion(outputs, labels)  # One number, the average batch loss
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(valloader)
        avg_acc_across_batches = (running_accuracy / len(valloader)) * 100

        print("Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%".format(avg_loss_across_batches, avg_acc_across_batches))
        print("***************************************************")

    # test
    model.train(False)
    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(testloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / batch_size
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(testloader)
    avg_acc_across_batches = (running_accuracy / len(testloader)) * 100

    print("Test Loss: {0:.3f}, Test Accuracy: {1:.1f}%".format(avg_loss_across_batches, avg_acc_across_batches))
    print("***************************************************")


hard_train(0.0005, optim.RMSprop, 20, 50, 0.23, 12)
