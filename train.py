import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import albumentations as A
from sklearn.metrics import f1_score

from dataset import SDSSData, SDSSData_train, SDSSData_test
from model import CNN_with_Unet

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ("galaxy", "star")
train_transform = A.Compose(
    [
        A.Rotate(limit=35, p=0.1),
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),
        A.Affine(shear=(-45, 45), p=0.1),
        A.OpticalDistortion(p=0.1),
        A.GaussNoise(p=0.5),
    ],
)


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

    return epoch_run_results


def validate_one_epoch(model, criterion, valloader):
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
            running_f1 += f1_score(labels.cpu(), torch.argmax(outputs.cpu(), dim=1))

    avg_loss = running_loss / len(valloader)
    avg_acc = (running_accuracy / len(valloader)) * 100
    avg_f1 = running_f1 / len(valloader)

    return avg_loss, avg_acc, avg_f1


def train_and_evaluate(params, model, trial):
    data = SDSSData(params["dist_from_center"], is_tunning=True)
    trainset = SDSSData_train(data, transform=params["transform"])

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params["learning_rate"])

    for epoch_index in range(params["num_epochs"]):
        train_one_epoch(model, optimizer, criterion, trainloader)

        val_loss, val_accuracy, val_f1 = validate_one_epoch(model, criterion, valloader)

    return val_accuracy, val_f1


def hard_train_and_test(params):
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
    trainset = SDSSData_train(data, transform=params["transform"])
    testset = SDSSData_test(data)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=params["batch_size"], shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=params["batch_size"], shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = params["optimizer"](model.parameters(), lr=params["learning_rate"])

    for epoch_index in range(params["num_epochs"]):
        print(f"Epoch: {epoch_index + 1}\n")

        epoch_run_results = train_one_epoch(model, optimizer, criterion, trainloader)

        for loss, accuracy, f1 in epoch_run_results:
            print(f"loss: {loss:.3f}, accuracy: {accuracy:.1f}%, f1: {f1:.1f}")

        val_loss, val_accuracy, val_f1 = validate_one_epoch(model, criterion, valloader)
        print(f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.1f}%, Val f1: {val_f1:.1f}")
        print("***************************************************")

    # test
    test_loss, test_accuracy, test_f1 = validate_one_epoch(model, criterion, testloader)
    print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.1f}%, Test f1: {test_f1:.1f}")
    print("***************************************************")


def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "dist_from_center": trial.suggest_int("dist_from_center", 10, 20),
        "batch_size": trial.suggest_categorical("batch_size", [50, 100, 150]),
        "drop_out": trial.suggest_float("drop_out", 0.1, 0.3),
        "num_epochs": trial.suggest_int("num_epochs", 5, 15),
        "hidden_nodes": trial.suggest_categorical("hidden_nodes", [256, 512, 1024]),
        "transform": train_transform,
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

    accuracy, f1 = train_and_evaluate(params, model, trial)

    return accuracy, f1


def tune_parameters(transform=None):
    if transform == None:
        study_name = f"maximizing-accuracy-f1"
    else:
        study_name = f"maximizing-accuracy-f1-transform"

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(),
        storage="sqlite:///results.db",  # just to show how man do paralellization, my machine too weak for paralellization *cry*
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    # params = {
    #     "learning_rate": 0.00172,
    #     "optimizer": optim.Adam,
    #     "dist_from_center": 20,
    #     "hidden_nodes": 512,
    #     "batch_size": 100,
    #     "drop_out": 0.27,
    #     "num_epochs": 2,
    #     "transform": train_transform,
    # }

    # hard_train_and_test(params)

    train_transform = None
    tune_parameters()
