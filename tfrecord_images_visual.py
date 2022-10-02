import tensorflow as tf
from pasta import augment

from tfrecords import get_num_of_images


def deserialization_fn(serialized_example):
    parsed_example = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
    image = tf.image.resize(image, size=(244, 244))
    label = tf.cast(parsed_example['image/class/label'], tf.int64)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label



def get_backbone_inference_dataset(tfrecord_paths, cache=False, repeat=False, shuffle=False, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
    data_len = sum([get_num_of_images(file) for file in tfrecord_paths])
    dataset = dataset.shuffle(data_len // 10) if shuffle else dataset
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    AUTO = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(deserialization_fn, num_parallel_calls=AUTO)

    return dataset

# backbone_iter_dataset_encode = get_backbone_inference_dataset(train_set_path, shuffle=True, augment=False)
