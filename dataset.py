from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def extract_data_from_coord(img, x, y, dist_from_center):
    return img[:, x - dist_from_center : x + dist_from_center, y - dist_from_center : y + dist_from_center]


def create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center=5):
    label = []
    data = []
    shift_x, shift_y = (
        13 + dist_from_center,
        4 + dist_from_center,
    )  # 13 and 4 recorded max shifts after processing

    for i in range(tensor_img.shape[0]):
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


class SDSSData:
    def __init__(self, dist_from_center=5, is_tunning=False):
        tensor_img = torch.load(f"./processed/img_tensor.pt")
        tensor_gal = torch.load(f"./processed/gal_tensor.pt")
        tensor_sta = torch.load(f"./processed/sta_tensor.pt")

        if is_tunning:  # use only 5 images for hyper-parameters tunning
            data, label = create_learning_data(tensor_img[:5], tensor_gal[:5], tensor_sta[:5], dist_from_center)
        else:
            data, label = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, stratify=label)

        self.train_data = train_data
        self.train_label = train_label

        tensor_img = torch.load(f"./processed/img_tensor_test_80.pt")
        tensor_gal = torch.load(f"./processed/gal_tensor_test_80.pt")
        tensor_sta = torch.load(f"./processed/sta_tensor_test_80.pt")

        data_80, label_80 = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

        self.test_data = torch.cat((test_data, data_80))
        self.test_label = torch.cat((test_label, label_80))


class SDSSData_train(Dataset):
    def __init__(self, data_origin, transform=None):
        self.data = data_origin.train_data
        self.label = data_origin.train_label

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            augmentations = self.transform(image=image.numpy())
            image = augmentations["image"]
        return torch.from_numpy(image), label


class SDSSData_test(Dataset):
    def __init__(self, data_origin):
        self.data = data_origin.test_data
        self.label = data_origin.test_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def test():
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(shear=(-45, 45), p=0.3),
            A.OpticalDistortion(p=0.5),
            A.GaussNoise(p=0.8),
        ],
    )
    data = SDSSData(10)

    trainset = SDSSData_train(data, transform=train_transform)
    testset = SDSSData_test(data)

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=500, shuffle=False, num_workers=2)

    for batch_index, data in enumerate(trainloader):
        inputs, labels = data[0], data[1]
        print(inputs.shape)
        print(labels.shape)


if __name__ == "__main__":
    test()
