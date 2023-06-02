from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb

def create_rbg(start_index, end_index):
    for index in range(start_index, end_index + 1):
        g_band = fits.open(f"./data_align/frame-g-008162-6-0{index:03d}.fits", ext=0)[0].data
        r_band = fits.open(f"./data_align/frame-r-008162-6-0{index:03d}.fits", ext=0)[0].data
        i_band = fits.open(f"./data_align/frame-i-008162-6-0{index:03d}.fits", ext=0)[0].data

        rgb_default = make_lupton_rgb(i_band, r_band, g_band, Q=10, stretch=0.5, filename=f"data_align/{index}.jpg")

        plt.figure()
        plt.axis("off")
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        plt.imshow(rgb_default, origin="lower")


def align_bands(img, wcs, coord):
    x, y = wcs.world_to_pixel(coord)
    shift_x = int(x[()] - img[0].header["CRPIX1"])
    shift_y = int(y[()] - img[0].header["CRPIX2"])

    img[0].data = np.roll(img[0].data, (shift_x, shift_y), axis=(1, 0))

    return img[0].data


def create_tensor_from_img(start_index, end_index, filepath, save_to_file, ref_band="g"):
    data = []
    bands = ["g", "r", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        channels = []
        ref_band_img = fits.open(f"./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits", ext=0)

        if(save_to_file):
            ref_band_img.writeto(f"./data_align/frame-{ref_band}-008162-6-0{i:03d}.fits", overwrite=True)

        channels.append(ref_band_img[0].data)

        wcs = WCS(ref_band_img[0].header)

        for band in bands:
            if band == ref_band:
                continue

            band_img = fits.open(f"./{filepath}/frame-{band}-008162-6-0{i:03d}.fits", ext=0)

            coord = SkyCoord(band_img[0].header["CRVAL1"], band_img[0].header["CRVAL2"], unit="deg")

            band_img[0].data = align_bands(band_img, wcs, coord)

            if(save_to_file):
                band_img.writeto(f"./data_align/frame-{band}-008162-6-0{i:03d}.fits", overwrite=True)

            channels.append(band_img[0].data)

            band_img.close()

        data.append(np.stack(channels, axis=2))
        ref_band_img.close()

    return np.stack(data, axis=0)


def create_tensor_from_coordinate(start_index, end_index, filepath, ref_band="g"):
    data = []
    bands = ["g", "r", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        channels = []
        ref_band_img = fits.open(f"./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits", ext=0)



        channels.append(ref_band_img[0].data)

        wcs = WCS(ref_band_img[0].header)

        for band in bands:
            if band == ref_band:
                continue

            band_img = fits.open(f"./{filepath}/frame-{band}-008162-6-0{i:03d}.fits", ext=0)

            coord = SkyCoord(band_img[0].header["CRVAL1"], band_img[0].header["CRVAL2"], unit="deg")

            band_img[0].data = align_bands(band_img, wcs, coord)


            channels.append(band_img[0].data)

            band_img.close()

        data.append(np.stack(channels, axis=2))
        ref_band_img.close()

    return np.stack(data, axis=0)


def save_tensor(data, filepath):
    np.save(f"{filepath}.npy", data)
