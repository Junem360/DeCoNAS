import os
import tensorflow as tf
import glob


def threaded_input_word_pipeline(base_dir,
                                 file_patterns,
                                 num_threads=4,
                                 batch_size=32,
                                 num_epochs=None,
                                 is_train=True,
                                 file_num=None):
    queue_capacity = num_threads * batch_size * 16
    # Allow a smaller final batch if we are going for a fixed number of epochs
    final_batch = (num_epochs is not None)

    data_queue, meta_data = _get_data_queue(base_dir,
                                 file_patterns,
                                 capacity=queue_capacity,
                                 num_epochs=num_epochs,
                                 is_train=is_train,
                                 file_num=file_num)

    image_ori, gt_label = _read_DIV2K_tfrecord(data_queue)
    images, labels = tf.train.shuffle_batch([image_ori, gt_label], batch_size=batch_size, capacity=queue_capacity, num_threads=num_threads,
                                            min_after_dequeue=10, allow_smaller_final_batch=final_batch)

    return images, labels, meta_data


def _read_DIV2K_tfrecord(data_queue):
    reader = tf.TFRecordReader()  # Construct a general reader
    _, example_serialized = reader.read(data_queue)

    feature_map = {
        'train/input_LR': tf.FixedLenFeature([], tf.string),
        'train/gt_HR': tf.FixedLenFeature([], tf.string),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    image = tf.decode_raw(features['train/input_LR'], tf.float32)
    label = tf.decode_raw(features['train/gt_HR'], tf.float32)
    image = tf.reshape(image, [64, 64, 3])
    label = tf.reshape(label, [64, 64, 3])

    return image, label


def _get_data_queue(base_dir, file_patterns=['*.tfrecords'], capacity=2 ** 15,
                    num_epochs=None, is_train=False, file_num=None):
    """Get a data queue for a list of record files"""
    # List of lists ...
    data_files = [tf.gfile.Glob(os.path.join(base_dir, file_pattern))
                  for file_pattern in file_patterns]
    meta_data = {}
    record_data_num = []

    # flatten
    data_files = [data_file for sublist in data_files for data_file in sublist]
    # if file_num is not None and file_num < len(data_files):
    #     data_files = data_files[0:file_num]
    for file_name in data_files:
        record_data_num.append(len([x for x in tf.python_io.tf_record_iterator(file_name)]))
    meta_data["data_nums"] = record_data_num
    meta_data["total_data_num"] = sum(record_data_num)
    meta_data["file_num"]= len(data_files)
    meta_data["data_files"] = data_files
    data_queue = tf.train.string_input_producer(data_files,
                                                capacity=capacity,
                                                shuffle=False,
                                                num_epochs=num_epochs)
    return data_queue, meta_data
