import os


def run_command(command):
    stream = os.popen(command)
    output = stream.read()
    print(output)


def get_data_galaxy_and_star(filepath):
    run_command(
        f"wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-gal.fits.gz -P ./{filepath}"
    )
    run_command(
        f"wget https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-star.fits.gz -P ./{filepath}"
    )


def get_data_images(start_index, end_index, filepath):
    f = open("spec-list.txt", "w")
    bands = ["r", "g", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        for band in bands:
            f.write(f"frame-{band}-008162-6-0{i:03d}.fits.bz2\n")
        f.write(f"frame-irg-008162-6-0{i:03d}.jpg\n")

    f.close()

    run_command(
        f'wget -i spec-list.txt -r --no-parent -nd -B "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/" -P ./{filepath}'
    )


def decompress(filepath):
    run_command(f"gzip -d {filepath}/*.gz")
    run_command(f"bzip2 -d {filepath}/*.bz2")
