import torch
import numpy as np

def extract_data_from_coord(img,x,y,dist_from_center):

    band_0 = torch.flatten(img[x-dist_from_center:x+dist_from_center,y-dist_from_center:y+dist_from_center,0])
    band_1 = torch.flatten(img[x-dist_from_center:x+dist_from_center,y-dist_from_center:y+dist_from_center,1])
    band_2 = torch.flatten(img[x-dist_from_center:x+dist_from_center,y-dist_from_center:y+dist_from_center,2])
    band_3 = torch.flatten(img[x-dist_from_center:x+dist_from_center,y-dist_from_center:y+dist_from_center,3])
    band_4 = torch.flatten(img[x-dist_from_center:x+dist_from_center,y-dist_from_center:y+dist_from_center,4])

    return torch.hstack((band_0,band_1,band_2,band_3,band_4))

def save_data(tensor, filepath):
    torch.save(tensor,f'{filepath}.pt')


def create_learning_data(tensor_img_path, tensor_gal_path, tensor_sta_path, dist_from_center=5):
    tensor_img = torch.load(f"./{tensor_img_path}.pt")
    tensor_gal = torch.load(f"./{tensor_gal_path}.pt").int()
    tensor_sta = torch.load(f"./{tensor_sta_path}.pt").int()

    label = []
    data = []
    shift_x, shift_y =(13 + dist_from_center,4 + dist_from_center)

    for i in range(tensor_img.shape[0]):
        selection_gal = torch.logical_and(tensor_gal[i,:,1]> shift_x ,tensor_gal[i,:,1]< 2048 - shift_x)
        selection_gal = torch.logical_and(selection_gal,tensor_gal[i,:,0]> shift_y)
        selection_gal = torch.logical_and(selection_gal,tensor_gal[i,:,0]< 1489 - shift_y)

        data += [extract_data_from_coord(tensor_img[i],coord[1],coord[0],dist_from_center) for coord in tensor_gal[i][selection_gal]]
        label += [0] * len(tensor_gal[i][selection_gal])

        selection_sta = torch.logical_and(tensor_sta[i,:,1]> shift_x ,tensor_sta[i,:,1]< 2048 - shift_x)
        selection_sta = torch.logical_and(selection_sta,tensor_sta[i,:,0]> shift_y)
        selection_sta = torch.logical_and(selection_sta,tensor_sta[i,:,0]< 1489 - shift_y)  

        data += [extract_data_from_coord(tensor_img[i],coord[1],coord[0],dist_from_center) for coord in tensor_sta[i][selection_sta]]
        label += [1] * len(tensor_sta[i][selection_sta])

    save_data(torch.vstack(data), 'data')
    save_data(torch.tensor(label), 'label')