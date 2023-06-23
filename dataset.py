from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import process


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
    def __init__(self, num_batch, dist_from_center=5):
        tensor_img = torch.load(f"./processed/img_tensor_0.pt")
        tensor_gal = torch.load(f"./processed/gal_tensor_0.pt")
        tensor_sta = torch.load(f"./processed/sta_tensor_0.pt")

        data, label = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

        for i in range(1, num_batch):
            tensor_img = torch.load(f"./processed/img_tensor_{i}.pt")
            tensor_gal = torch.load(f"./processed/gal_tensor_{i}.pt")
            tensor_sta = torch.load(f"./processed/sta_tensor_{i}.pt")

            data_1, label_1 = create_learning_data(tensor_img, tensor_gal, tensor_sta, dist_from_center)

            data = torch.cat((data, data_1))
            label = torch.cat((label, label_1))

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
        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return self.data[index], self.label[index]


class SDSSData_test(Dataset):
    def __init__(self, data_origin, transform=None):
        self.data = data_origin.test_data
        self.label = data_origin.test_label

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
    data = SDSSData(1, 10)

    trainset = SDSSData_train(data)
    testset = SDSSData_test(data)

    train_iter = iter(trainset)
    image, label = next(train_iter)
    print(image.shape)
    print(f"train size: {len(trainset)}")

    train_iter = iter(testset)
    image, label = next(train_iter)
    print(image.shape)
    print(f"test size: {len(testset)}")

    trainset, valset = torch.utils.data.random_split(trainset, [len(trainset) - len(trainset) // 2, len(trainset) // 2])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False, num_workers=2)


if __name__ == "__main__":
    test()
