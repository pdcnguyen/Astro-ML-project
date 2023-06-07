import acquire
import process

# acquire.get_data_galaxy_and_star('data')
# acquire.get_data_images(90,120, 'data')
# acquire.decompress()


data = process.create_img_tensor(90,100,'data', ref_band="r")
process.save_tensor(data,'img_tensor')

data_star, data_gal = process.create_star_gal_tensor(90,100,'data',ref_band="r")
process.save_tensor(data_star,'star_tensor')
process.save_tensor(data_gal,'gal_tensor')