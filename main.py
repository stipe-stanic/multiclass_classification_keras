import os
import numpy as np
import tensorflow as tf
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# dataset of records from one or more TFRecord files
raw_dataset = tf.data.TFRecordDataset('data/img_products/guie-products10k-train-00-7097.tfrec')
print(type(raw_dataset))

for i, raw_record in enumerate(raw_dataset.take(3)):
    example = tf.train.Example()  # object storing data for training and inference
    example.ParseFromString(raw_record.numpy())  # decoding the message
    print(example)
    info = {}
    for key, val in example.features.feature.items():
        if key == "image/encoded":
            info[key] = val.bytes_list.value[0]
        elif key == "image/class/label":
            info[key] = val.int64_list.value[0]
    img_arr = np.frombuffer(info["image/encoded"], dtype=np.uint8)

    img = Image.fromarray(img_arr)
    img.save(f'data/img_products/guie-products10k-train-00-7097.tfrec.{str(i).zfill(5)}.png')




# if __name__ == '__main__':
