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