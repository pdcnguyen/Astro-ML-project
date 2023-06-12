import acquire
import process
import prepare

# acquire.get_data_galaxy_and_star('data')
# acquire.get_data_images(80,89, 'data')
# acquire.decompress('data')


data = process.create_img_tensor(80,120,'data', ref_band="r")
process.save_tensor(data,'img_tensor')

data_star, data_gal = process.create_star_gal_tensor(80,120,'data',ref_band="r")
process.save_tensor(data_star,'star_tensor')
process.save_tensor(data_gal,'gal_tensor')

prepare.create_learning_data('img_tensor','star_tensor','gal_tensor',7)