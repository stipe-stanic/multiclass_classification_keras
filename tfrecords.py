import tensorflow as tf
import random
# import zipfile


def get_num_of_images(file):
    return int(file.split('/')[-1].split('.')[0].split('-')[-1])


# zip_file = "data/guie-imagenet1k-mini1-tfrecords-label-0-999.zip"
# zip_file_xtract_to_dir = "data/image_net_1k"
# if zipfile.is_zipfile(zip_file):
#     with zipfile.ZipFile(zip_file, 'r') as zip_obj:
#         zip_obj.extractall(zip_file_xtract_to_dir)
#         print('Folder is unzipped')

files_path = "data/image_net_1k/guie-imagenet1k"
train_shard_suffix = '*-train-*.tfrec'

train_set_path = []
valid_set_path = []

files = sorted(tf.io.gfile.glob(files_path + f'{train_shard_suffix}'))

train_set_path += random.sample(files, int(len(files) * 0.9))
valid_set_path += [file for file in files if file not in train_set_path]

print(files_path, ", number of tfrecords = ", len(files))

train_set_path = sorted(train_set_path)
valid_set_path = sorted(valid_set_path)

train_set_len = sum([get_num_of_images(file) for file in train_set_path])
valid_set_len = sum([get_num_of_images(file) for file in valid_set_path])

print("Number of images: ", train_set_len)
print("Number of images: ", valid_set_len)



















