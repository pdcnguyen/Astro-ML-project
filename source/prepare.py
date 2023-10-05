import os
import process

from config import START_INDEX, STOP_INDEX, TEST_FILES_lIST


def run_command(command):
    stream = os.popen(command)
    output = stream.read()
    print(output)


def get_data(index_list, filepath):
    run_command(
        f"wget -nc https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-gal.fits.gz -P ./{filepath}"
    )
    run_command(
        f"wget -nc https://data.sdss.org/sas/dr17/eboss/sweeps/dr13_final/301/calibObj-008162-6-star.fits.gz -P ./{filepath}"
    )

    run_command("touch file-list.txt")
    file = open("file-list.txt", mode="w", encoding="utf-8")
    bands = ["r", "g", "i", "u", "z"]

    for i in index_list:
        for band in bands:
            file.write(f"frame-{band}-008162-6-0{i:03d}.fits.bz2\n")

    file.close()

    run_command(
        f'wget -i file-list.txt -r --no-parent -nd -B "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/6/" -nc -P ./{filepath}'
    )


def prepate_data(start_index, stop_index, test_index_list):
    raw_data_path = "./data/raw"
    processed_data_path = "./data/processed"

    tensor_img_path = f"{processed_data_path}/img_tensor"
    tensor_gal_path = f"{processed_data_path}/gal_tensor"
    tensor_sta_path = f"{processed_data_path}/sta_tensor"

    if not os.path.exists("./data"):
        os.mkdir("./data")
        os.mkdir(raw_data_path)
        os.mkdir(processed_data_path)

    # ============== getting data ===============

    index_list = list(range(start_index, stop_index + 1))
    get_data(index_list, "data/raw")

    # ============== align bands and create tensor of train files ===============

    train_index_list = [i for i in index_list if i not in test_index_list]

    train_index_list_chunks = [
        train_index_list[i : i + 40] for i in range(0, len(train_index_list), 40)
    ]

    for i, index_list in enumerate(train_index_list_chunks):
        img_tensor_train = process.create_img_tensor(
            index_list, "data/raw", ref_band="r"
        )
        process.save_tensor(img_tensor_train, f"{tensor_img_path}_{i}")

        star_tensor_train, gal_tensor_train = process.create_star_gal_tensor(
            index_list, "data/raw", ref_band="r"
        )
        process.save_tensor(star_tensor_train, f"{tensor_sta_path}_{i}")
        process.save_tensor(gal_tensor_train, f"{tensor_gal_path}_{i}")

    # ============== align bands and create tensor of test files===============

    img_tensor_test = process.create_img_tensor(
        test_index_list, "data/raw", ref_band="r"
    )
    process.save_tensor(img_tensor_test, f"{tensor_img_path}_test")

    star_tensor_test, gal_tensor_test = process.create_star_gal_tensor(
        test_index_list, "data/raw", ref_band="r"
    )
    process.save_tensor(star_tensor_test, f"{tensor_sta_path}_test")
    process.save_tensor(gal_tensor_test, f"{tensor_gal_path}_test")


if __name__ == "__main__":
    prepate_data(START_INDEX, STOP_INDEX, TEST_FILES_lIST)
