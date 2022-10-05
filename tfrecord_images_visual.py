import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tfrecords import train_set_path
from tfrecords import get_num_of_images
from mpl_toolkits import axes_grid1
from pasta import augment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # info, warning and error messages are omitted


def arcface_format(image, label_group):
    return {'inp1': image, 'inp2': label_group}, label_group  # using key-value pairs to reference Tensors


def rescale_image(image, label_group):
    image = tf.cast(image, tf.float32) * 255.0
    return image, label_group


def deserialization_fn(serialized_example):  # param is scalar string tensor
    # returns a dict mapping feature keys to Tensor values
    parsed_example = tf.io.parse_single_example(
        serialized_example,
        features={
            # configuration for parsing a fixed-length input feature.
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
    )

    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)  # decodes image into a uint8 tensor.
    image = tf.image.resize(image, size=(244, 244))  # 1-D int32 Tensor
    label = tf.cast(parsed_example['image/class/label'], tf.int64)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def get_backbone_inference_dataset(tfrecord_paths, cache=False, repeat=False, shuffle=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)  # creates a dataset
    data_len = sum([get_num_of_images(file) for file in tfrecord_paths])
    dataset = dataset.shuffle(data_len // 10) if shuffle else dataset
    dataset = dataset.flat_map(tf.data.TFRecordDataset)  # order of the dataset stays the same

    AUTO = tf.data.experimental.AUTOTUNE  # prompts tf.data runtime to tune the value dynamically at runtime

    # transforms each element of the dataset, the number of parallel calls is set dynamically based on available CPU.
    dataset = dataset.map(deserialization_fn, num_parallel_calls=AUTO)
    dataset = dataset.map(rescale_image, num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)

    dataset = dataset.batch(200)
    dataset = dataset.prefetch(AUTO)  # prefetches elements from the dataset

    return dataset


backbone_infer_dataset_encode = get_backbone_inference_dataset(train_set_path, shuffle=True, augment=False)

num_cols = 3
num_rows = 5

# slices batched tensor into smaller, unbatched tensors
backbone_infer_dataset_encode = backbone_infer_dataset_encode.unbatch().batch(num_cols * num_rows)
x, y = next(iter(backbone_infer_dataset_encode))  # create object_iterator and returns items in iterator
print(x['inp1'].shape)  # (batch, height, width, rgb)

fig = plt.figure(figsize=(10, 10))
grid = axes_grid1.ImageGrid(fig, 111, nrows_ncols=(num_cols, num_rows), axes_pad=0.1)  # creates grid

for i, ax in enumerate(grid):  # adds counter to an iterable
    ax.imshow(x['inp1'][i]/255)
    ax.axis("off")

plt.show()

del backbone_infer_dataset_encode  # deleting the Tensor
