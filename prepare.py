import acquire
import process
import os

tensor_img_path = "./processed/img_tensor_1"
tensor_gal_path = "./processed/gal_tensor_1"
tensor_sta_path = "./processed/sta_tensor_1"

start_index = 160
stop_index = 238

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


# ============== align bands and create tensor ===============
img_tensor = process.create_img_tensor(start_index, stop_index, "data", ref_band="r")
process.save_tensor(img_tensor, tensor_img_path)

star_tensor, gal_tensor = process.create_star_gal_tensor(start_index, stop_index, "data", ref_band="r")
process.save_tensor(star_tensor, tensor_sta_path)
process.save_tensor(gal_tensor, tensor_gal_path)
