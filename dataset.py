import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch


def extract_data_from_coord(img, x, y, dist_from_center):
    return img[:, x - dist_from_center : x + dist_from_center, y - dist_from_center : y + dist_from_center]


def create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center=5):
    label = []
    data = []
    shift_x, shift_y = (
        13 + dist_from_center,
        4 + dist_from_center,
    )  # 13 and 4 recorded max shifts after processing

    for i in range(1):
        selection_gal = torch.logical_and(tensor_gal[i, :, 1] > shift_x, tensor_gal[i, :, 1] < 2048 - shift_x)
        selection_gal = torch.logical_and(selection_gal, tensor_gal[i, :, 0] > shift_y)
        selection_gal = torch.logical_and(selection_gal, tensor_gal[i, :, 0] < 1489 - shift_y)

        data += [
            extract_data_from_coord(tensor_img[i], coord[1], coord[0], dist_from_center)
            for coord in tensor_gal[i][selection_gal]
        ]
        label += [0] * len(tensor_gal[i][selection_gal])  # galaxy as 0

        selection_sta = torch.logical_and(tensor_sta[i, :, 1] > shift_x, tensor_sta[i, :, 1] < 2048 - shift_x)
        selection_sta = torch.logical_and(selection_sta, tensor_sta[i, :, 0] > shift_y)
        selection_sta = torch.logical_and(selection_sta, tensor_sta[i, :, 0] < 1489 - shift_y)

        data += [
            extract_data_from_coord(tensor_img[i], coord[1], coord[0], dist_from_center)
            for coord in tensor_sta[i][selection_sta]
        ]
        label += [1] * len(tensor_sta[i][selection_sta])  # star as 1

    return torch.stack(data), torch.tensor(label)


class SDSSData(Dataset):
    def __init__(
        self,
        tensor_img_path,
        tensor_gal_path,
        tensor_sta_path,
        dist_from_center=5,
        transform=None,
    ):
        tensor_img = torch.load(f"./processed/img_tensor_0.pt")
        tensor_gal = torch.load(f"./processed/gal_tensor_0.pt").int()
        tensor_sta = torch.load(f"./processed/sta_tensor_0.pt").int()

        data_0, label_0 = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

        tensor_img = torch.load(f"./processed/img_tensor_1.pt")
        tensor_gal = torch.load(f"./processed/gal_tensor_1.pt").int()
        tensor_sta = torch.load(f"./processed/sta_tensor_1.pt").int()

        data_1, label_1 = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

        self.data = torch.cat((data_0, data_1))
        self.label = torch.cat((label_0, label_1))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return self.data[index], self.label[index]


def test():
    tensor_img_path = "./processed/img_tensor"
    tensor_gal_path = "./processed/gal_tensor"
    tensor_sta_path = "./processed/sta_tensor"

    trainset = SDSSData(tensor_img_path, tensor_gal_path, tensor_sta_path, 10)

    train_iter = iter(trainset)

    image, label = next(train_iter)

    print(image.shape)
    print(len(trainset))
    print(label)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False, num_workers=2)


if __name__ == "__main__":
    test()
