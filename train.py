import torch

# import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import SDSSData
from model import CNN_with_Unet

torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 20
dist_from_center = 20
learning_rate = 0.0001
num_epochs = 100
tensor_img_path = "./processed/img_tensor"
tensor_gal_path = "./processed/gal_tensor"
tensor_sta_path = "./processed/sta_tensor"


trainset = SDSSData(tensor_img_path, tensor_gal_path, tensor_sta_path, dist_from_center)

classes = ("galaxy", "star")

trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=batch_size, shuffle=False, num_workers=2
# )


net = CNN_with_Unet(
    in_channels=5,
    out_channels=1,
    num_of_class=len(classes),
    dist_from_center=dist_from_center,
)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def train_one_epoch():
    net.train(True)

    running_loss = 0.0
    running_accuracy = 0.0

    for batch_index, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)  # shape: [batch_size, 10]
        correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        running_accuracy += correct / batch_size

        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 9:  # print every 10 batches
            avg_loss_across_batches = running_loss / 10
            avg_acc_across_batches = (running_accuracy / 10) * 100
            print(
                "Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%".format(
                    batch_index + 1, avg_loss_across_batches, avg_acc_across_batches
                )
            )
            running_loss = 0.0
            running_accuracy = 0.0


def validate_one_epoch():
    net.train(False)
    running_loss = 0.0
    running_accuracy = 0.0

    for i, data in enumerate(valloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(inputs)  # shape: [batch_size, 10]
            correct = torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            running_accuracy += correct / batch_size
            loss = criterion(outputs, labels)  # One number, the average batch loss
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(valloader)
    avg_acc_across_batches = (running_accuracy / len(valloader)) * 100

    print("Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%".format(avg_loss_across_batches, avg_acc_across_batches))
    print("***************************************************")


for epoch_index in range(num_epochs):
    print(f"Epoch: {epoch_index + 1}\n")

    train_one_epoch()
    validate_one_epoch()


print("Finished Training")
