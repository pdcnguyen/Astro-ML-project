from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import make_lupton_rgb


def align_bands(start_index, end_index, ref_band="g"):
    bands = ["g", "r", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        ref_band_img = fits.open(
            f"./data/frame-{ref_band}-008162-6-0{i:03d}.fits", ext=0
        )
        wcs = WCS(ref_band_img[0].header)

        coord = SkyCoord(
            ref_band_img[0].header["CRVAL1"],
            ref_band_img[0].header["CRVAL2"],
            unit="deg",
        )

        ref_band_img.writeto(
            f"./data_align/frame-{ref_band}-008162-6-0{i:03d}.fits", overwrite=True
        )

        for band in bands:
            if(band == ref_band):
                continue

            band_img = fits.open(f"./data/frame-{band}-008162-6-0{i:03d}.fits", ext=0)

            coord = SkyCoord(
                band_img[0].header["CRVAL1"], band_img[0].header["CRVAL2"], unit="deg"
            )
            x, y = wcs.world_to_pixel(coord)
            shift_x = int(x[()] - band_img[0].header["CRPIX1"])
            shift_y = int(y[()] - band_img[0].header["CRPIX2"])

            band_img[0].data = np.roll(band_img[0].data, (shift_x, shift_y), axis=(1, 0))

            band_img.writeto(
                f"./data_align/frame-{band}-008162-6-0{i:03d}.fits", overwrite=True
            )
            band_img.close()

def create_rbg(start_index, end_index):
    for index in range(start_index, end_index + 1):
        g_band = fits.open(f'./data_align/frame-g-008162-6-0{index:03d}.fits', ext=0)[0].data
        r_band = fits.open(f'./data_align/frame-r-008162-6-0{index:03d}.fits', ext=0)[0].data
        i_band = fits.open(f'./data_align/frame-i-008162-6-0{index:03d}.fits', ext=0)[0].data

        rgb_default = make_lupton_rgb(i_band, r_band, g_band, Q=10, stretch=0.5, filename=f"data_align/{index}.jpg")

        plt.figure()
        plt.axis("off")
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        plt.imshow(rgb_default, origin='lower')

align_bands(100,119)
create_rbg(100,119)
