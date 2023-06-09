from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb
import torch

def create_rbg(g_band, r_band, i_band):
    rgb_default = make_lupton_rgb(i_band, r_band, g_band, Q=10, stretch=0.5)

    plt.figure()
    plt.axis("off")
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    plt.imshow(rgb_default, origin="lower")


def align_bands(img, wcs, coord):
    x, y = wcs.world_to_pixel(coord)
    shift_x = int(x[()] - img[0].header["CRPIX1"])
    shift_y = int(y[()] - img[0].header["CRPIX2"])

    img[0].data = np.roll(img[0].data, (shift_x, shift_y), axis=(1, 0))

    print(shift_x, shift_y)

    return img[0].data


def pad_array(array, shape):
    padding_array = np.zeros(shape)
    padding_array[:array.shape[0],:array.shape[1]] = array

    return padding_array


def create_img_tensor(start_index, end_index, filepath, ref_band="r"):
    data = []
    bands = ["g", "r", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        channels = []

        ref_band_img = fits.open(f"./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits", ext=0)

        wcs = WCS(ref_band_img[0].header)

        for band in bands:
            if band == ref_band:
                channels.append(ref_band_img[0].data)
                continue

            band_img = fits.open(f"./{filepath}/frame-{band}-008162-6-0{i:03d}.fits", ext=0)

            coord = SkyCoord(band_img[0].header["CRVAL1"], band_img[0].header["CRVAL2"], unit="deg")

            band_img[0].data = align_bands(band_img, wcs, coord)

            channels.append(band_img[0].data)

            band_img.close()

        data.append(np.stack(channels, axis=2))
        ref_band_img.close()

    return np.stack(data, axis=0)


def create_star_gal_tensor(start_index, end_index, filepath, ref_band="r"):
    stars = fits.open(f'./{filepath}/calibObj-008162-6-star.fits', ext=0)
    gals = fits.open(f'./{filepath}/calibObj-008162-6-gal.fits', ext=0)

    stars_RA = np.array(stars[1].data.field('RA'))
    stars_DEC = np.array(stars[1].data.field('DEC'))
    stars_FIELD = np.array(stars[1].data.field('FIELD'))

    gals_RA = np.array(gals[1].data.field('RA'))
    gals_DEC = np.array(gals[1].data.field('DEC'))
    gals_FIELD = np.array(gals[1].data.field('FIELD'))

    data_stars = []
    data_gals = []

    max_shape_stars = (0,0)
    max_shape_gals = (0,0)

    for i in range(start_index,end_index+1):

        band = fits.open(f'./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits', ext=0)
        wcs = WCS(band[0].header)

        x_star, y_star = wcs.all_world2pix(stars_RA[stars_FIELD==i], stars_DEC[stars_FIELD==i], 0)
        x_gal, y_gal = wcs.all_world2pix(gals_RA[gals_FIELD==i], gals_DEC[gals_FIELD==i], 0)

        coord_stars = np.stack([x_star, y_star], axis=1)
        coord_gals = np.stack([x_gal, y_gal], axis=1)

        if(max_shape_stars[0] < coord_stars.shape[0]):
            max_shape_stars = coord_stars.shape

        if(max_shape_gals[0] < coord_gals.shape[0]):
            max_shape_gals = coord_gals.shape

        data_stars.append(coord_stars)
        data_gals.append(coord_gals)


    data_stars = [pad_array(star, max_shape_stars) for star in data_stars]
    data_gals = [pad_array(gal, max_shape_gals) for gal in data_gals]

    return np.stack(data_stars, axis=0), np.stack(data_gals, axis=0)


def save_tensor(data, filepath):
    t = torch.from_numpy(data)
    torch.save(t,f'{filepath}.pt')
