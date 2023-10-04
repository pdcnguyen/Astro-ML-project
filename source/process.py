from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from tqdm import tqdm
import numpy as np
import torch


# from astropy.visualization import make_lupton_rgb
# import matplotlib.pyplot as plt


# def create_rbg(g_band, r_band, i_band):
#     rgb_default = make_lupton_rgb(i_band, r_band, g_band, Q=10, stretch=0.5)

#     plt.figure()
#     plt.axis("off")
#     plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
#     plt.imshow(rgb_default, origin="lower")


def align_bands(img, wcs, coord):
    x, y = wcs.world_to_pixel(coord)
    shift_x = int(x[()] - img[0].header["CRPIX1"])
    shift_y = int(y[()] - img[0].header["CRPIX2"])

    img[0].data = np.roll(img[0].data, (shift_x, shift_y), axis=(1, 0))

    return img[0].data


def create_img_tensor(index_list, filepath, ref_band="r"):
    data = []
    bands = ["g", "r", "i", "u", "z"]

    for i in tqdm(index_list):
        channels = []

        ref_band_img = fits.open(
            f"./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits.bz2", ext=0
        )

        wcs = WCS(ref_band_img[0].header)

        for band in bands:
            if band == ref_band:
                channels.append(
                    torch.from_numpy(ref_band_img[0].data.astype(np.float32))
                )
                continue

            band_img = fits.open(
                f"./{filepath}/frame-{band}-008162-6-0{i:03d}.fits.bz2", ext=0
            )

            coord = SkyCoord(
                band_img[0].header["CRVAL1"], band_img[0].header["CRVAL2"], unit="deg"
            )

            band_img[0].data = align_bands(band_img, wcs, coord)

            channels.append(torch.from_numpy(band_img[0].data.astype(np.float32)))

            band_img.close()

        data.append(torch.stack(channels))
        ref_band_img.close()

    tensor = torch.stack(data)

    return tensor


def create_star_gal_tensor(index_list, filepath, ref_band="r"):
    stars = fits.open(f"./{filepath}/calibObj-008162-6-star.fits.gz", ext=0)
    gals = fits.open(f"./{filepath}/calibObj-008162-6-gal.fits.gz", ext=0)

    stars_RA = np.array(stars[1].data.field("RA"))
    stars_DEC = np.array(stars[1].data.field("DEC"))
    stars_FIELD = np.array(stars[1].data.field("FIELD"))

    gals_RA = np.array(gals[1].data.field("RA"))
    gals_DEC = np.array(gals[1].data.field("DEC"))
    gals_FIELD = np.array(gals[1].data.field("FIELD"))

    data_stars = []
    data_gals = []

    for i in index_list:
        band = fits.open(
            f"./{filepath}/frame-{ref_band}-008162-6-0{i:03d}.fits.bz2", ext=0
        )
        wcs = WCS(band[0].header)

        x_star, y_star = wcs.all_world2pix(
            stars_RA[stars_FIELD == i], stars_DEC[stars_FIELD == i], 0
        )
        x_gal, y_gal = wcs.all_world2pix(
            gals_RA[gals_FIELD == i], gals_DEC[gals_FIELD == i], 0
        )

        coord_stars = np.stack([x_star, y_star], axis=1).astype(int)
        coord_gals = np.stack([x_gal, y_gal], axis=1).astype(int)

        data_stars.append(torch.from_numpy(coord_stars))
        data_gals.append(torch.from_numpy(coord_gals))

    return data_stars, data_gals


def save_tensor(data, filepath):
    torch.save(data, f"{filepath}.pt")
