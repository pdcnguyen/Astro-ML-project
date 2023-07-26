import process as process
import os


def run_command(command):
    stream = os.popen(command)
    output = stream.read()
    print(output)


def get_data(start_index, end_index, filepath):
    run_command(
        f"wget -nc https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-gal.fits.gz -P ./{filepath}"
    )
    run_command(
        f"wget -nc https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-star.fits.gz -P ./{filepath}"
    )

    f = open("spec-list.txt", "w")
    bands = ["r", "g", "i", "u", "z"]

    for i in range(start_index, end_index + 1):
        for band in bands:
            f.write(f"frame-{band}-008162-6-0{i:03d}.fits.bz2\n")

    f.close()

    run_command(
        f'wget -i spec-list.txt -r --no-parent -nd -B "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/" -nc -P ./{filepath}'
    )


def decompress(filepath):
    run_command(f"gzip -d {filepath}/*.gz")
    run_command(f"bzip2 -d {filepath}/*.bz2")


def prepate_data(start_index, stop_index):
    tensor_img_path = "./processed/img_tensor"
    tensor_gal_path = "./processed/gal_tensor"
    tensor_sta_path = "./processed/sta_tensor"

    isExist = os.path.exists("./data")
    if not isExist:
        os.mkdir("./data")

    isExist = os.path.exists("./processed")
    if not isExist:
        os.mkdir("./processed")

    # ============== getting data and unpack ===============

    get_data(start_index, stop_index, "data")
    decompress("data")

    # ============== align bands and create tensor into parts, spare my poor machine ===============

    img_tensor = process.create_img_tensor(start_index, stop_index, "data", ref_band="r")
    process.save_tensor(img_tensor, f"{tensor_img_path}")

    star_tensor, gal_tensor = process.create_star_gal_tensor(start_index, stop_index, "data", ref_band="r")
    process.save_tensor(star_tensor, f"{tensor_sta_path}")
    process.save_tensor(gal_tensor, f"{tensor_gal_path}")

    # ============== align bands and create tensor of pic 80===============

    img_tensor_80 = process.create_img_tensor(80, 81, "data", ref_band="r", test_80=True)
    process.save_tensor(img_tensor_80, f"{tensor_img_path}_test_80")

    star_tensor_80, gal_tensor_80 = process.create_star_gal_tensor(80, 81, "data", ref_band="r", test_80=True)
    process.save_tensor(star_tensor_80, f"{tensor_sta_path}_test_80")
    process.save_tensor(gal_tensor_80, f"{tensor_gal_path}_test_80")


if __name__ == "__main__":
    prepate_data(80, 130)