import acquire
import process

# acquire.get_data_galaxy_and_star('data')
# acquire.get_data_images(90,120, 'data')
# acquire.decompress()


data = process.create_tensor_from_img(90,120,'data')
process.save_tensor(data,'tensor_img')