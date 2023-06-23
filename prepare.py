import acquire
import process
import os


def prepate_data():
    tensor_img_path = "./processed/img_tensor"
    tensor_gal_path = "./processed/gal_tensor"
    tensor_sta_path = "./processed/sta_tensor"

    start_index = 80  # 80
    stop_index = 237  # 237
    batch_size = 40

    isExist = os.path.exists("./data")
    if not isExist:
        os.mkdir("./data")

    isExist = os.path.exists("./processed")
    if not isExist:
        os.mkdir("./processed")

    # ============== getting data and unpack ===============

    # acquire.get_data_galaxy_and_star("data")
    # acquire.get_data_images(start_index, stop_index, "data")
    # acquire.decompress("data")

    # ============== align bands and create tensor into parts, spare my poor machine ===============

    for i in range((stop_index - start_index) // batch_size + 1):
        start = start_index + (i * batch_size)
        stop = start_index + (i + 1) * batch_size
        if start == stop_index:
            continue
        if stop > stop_index:
            stop = stop_index

        img_tensor = process.create_img_tensor(start_index + (i * batch_size), stop, "data", ref_band="r")
        process.save_tensor(img_tensor, f"{tensor_img_path}_{i}")

        star_tensor, gal_tensor = process.create_star_gal_tensor(
            start_index + (i * batch_size), stop, "data", ref_band="r"
        )
        process.save_tensor(star_tensor, f"{tensor_sta_path}_{i}")
        process.save_tensor(gal_tensor, f"{tensor_gal_path}_{i}")

    # ============== align bands and create tensor of pic 80===============

    img_tensor_80 = process.create_img_tensor(80, 81, "data", ref_band="r", test_80=True)
    process.save_tensor(img_tensor_80, f"{tensor_img_path}_test_80")

    star_tensor_80, gal_tensor_80 = process.create_star_gal_tensor(80, 81, "data", ref_band="r", test_80=True)
    process.save_tensor(star_tensor_80, f"{tensor_sta_path}_test_80")
    process.save_tensor(gal_tensor_80, f"{tensor_gal_path}_test_80")


if __name__ == "__main__":
    prepate_data()
