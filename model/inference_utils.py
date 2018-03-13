import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import pandas as pd
import numpy as np

MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
layer_to_extract = 'pool_3:0'
pretrain_model_name = 'classify_image_graph_def.pb'


def create_graph(pretrain_dir):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            pretrain_dir, pretrain_model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def maybe_download_and_extract(pretrain_dir):
    """Download and extract model tar file."""
    dest_directory = pretrain_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = MODEL_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(MODEL_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def extract_features(image_dir, pretrain_dir):
    if not os.path.exists(image_dir):
        print("image_dir does not exit!")
        return None

    maybe_download_and_extract(pretrain_dir)

    create_graph(pretrain_dir)

    with tf.Session() as sess:
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        final_array = []
        extract_tensor = sess.graph.get_tensor_by_name(layer_to_extract)
        counter = 0
        print("There are total " + str(len(os.listdir(image_dir))) + " images to process.")
        all_image_names = os.listdir(image_dir)
        all_image_names = pd.DataFrame({'file_name': all_image_names})

        for img in all_image_names['file_name'].values:
            temp_path = os.path.join(image_dir, img)

            image_data = tf.gfile.FastGFile(temp_path, 'rb').read()

            predictions = sess.run(extract_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            final_array.append(predictions)

        final_array = np.array(final_array)
    return final_array, all_image_names


def extract_image_id(image_name):
    name, extension = image_name.split(".")
    assert (extension == "jpg")
    return int(name.split("_")[-1])