from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


bands = ["r", "g", "i", "u", "z"]

for i in bands:
    hdul = fits.open(f'./data/frame-{i}-008162-6-0100.fits')
    print(WCS(hdul[0].header))
    image_data = fits.getdata(image_file, ext=0)
    hdul.close()


