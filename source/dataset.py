import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F


def extract_data_and_normalize(img, x, y, dist_from_center):
    cut_out = img[
        :,
        x - dist_from_center : x + dist_from_center,
        y - dist_from_center : y + dist_from_center,
    ]
    cut_out = torch.stack([F.normalize(a) for a in cut_out])

    return cut_out


def create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center=15):
    label = []
    data = []
    shift_x, shift_y = (
        13 + dist_from_center,
        4 + dist_from_center,
    )  # 13 and 4 recorded max shifts after processing

    for i in range(tensor_img.shape[0]):
        for j in [0, 1]:  # decode galaxy as 0, star as 1
            coord_list = tensor_gal[i] if j == 0 else tensor_sta[i]

            # remove coord that might not have sufficient data
            selection = torch.logical_and(
                coord_list[:, 1] > shift_x, coord_list[:, 1] < 2048 - shift_x
            )
            selection = torch.logical_and(selection, coord_list[:, 0] > shift_y)
            selection = torch.logical_and(selection, coord_list[:, 0] < 1489 - shift_y)

            filtered_coord = coord_list[selection]

            data += [
                extract_data_and_normalize(
                    tensor_img[i], coord[1], coord[0], dist_from_center
                )
                for coord in filtered_coord
            ]
            label += [j] * len(filtered_coord)

    return torch.stack(data), torch.tensor(label)


class SDSSData:
    def __init__(self, dist_from_center=15, is_tunning=False):
        data = []
        label = []

        if is_tunning:  # use only 4 images for hyper-parameters tunning
            tensor_img = torch.load("./data/processed/img_tensor_0.pt")
            tensor_gal = torch.load("./data/processed/gal_tensor_0.pt")
            tensor_sta = torch.load("./data/processed/sta_tensor_0.pt")

            extracted_data = create_learning_data(
                tensor_img[:4], tensor_gal[:4], tensor_sta[:4], dist_from_center
            )
            data.append(extracted_data[0])
            label.append(extracted_data[1])
        else:
            i = 0
            while os.path.isfile(f"./data/processed/img_tensor_{i}.pt"):
                tensor_img = torch.load(f"./data/processed/img_tensor_{i}.pt")
                tensor_gal = torch.load(f"./data/processed/gal_tensor_{i}.pt")
                tensor_sta = torch.load(f"./data/processed/sta_tensor_{i}.pt")

                extracted_data = create_learning_data(
                    tensor_img, tensor_gal, tensor_sta, dist_from_center
                )

                data.append(extracted_data[0])
                label.append(extracted_data[1])
                i += 1

        data = torch.cat(data)
        label = torch.cat(label)

        train_data, val_data, train_label, val_label = train_test_split(
            data, label, test_size=0.2, stratify=label
        )

        self.train_data = train_data
        self.train_label = train_label

        self.val_data = val_data
        self.val_label = val_label


class SDSSData_train(Dataset):
    def __init__(self, data_origin, transform=None):
        self.data = data_origin.train_data
        self.label = data_origin.train_label

        class_counts = data_origin.train_label.unique(return_counts=True)[1]
        self.sample_weights = [1 / class_counts[i] for i in data_origin.train_label]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            augmentations = self.transform(image=image.numpy())
            image = torch.from_numpy(augmentations["image"])
        return image, label


class SDSSData_val(Dataset):
    def __init__(self, data_origin, transform=None):
        self.data = data_origin.val_data
        self.label = data_origin.val_label

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            augmentations = self.transform(image=image.numpy())
            image = torch.from_numpy(augmentations["image"])
        return image, label


class SDSSData_test(Dataset):
    def __init__(self, dist_from_center):
        tensor_img = torch.load("./data/processed/img_tensor_test.pt")
        tensor_gal = torch.load("./data/processed/gal_tensor_test.pt")
        tensor_sta = torch.load("./data/processed/sta_tensor_test.pt")

        self.data, self.label = create_learning_data(
            tensor_img, tensor_gal, tensor_sta, dist_from_center
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
