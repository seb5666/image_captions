"""Predict captions on test images using trained model, with beam search method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import configuration
from ShowAndTellModel import build_model
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url, write_text_on_image
import numpy as np
from scipy.misc import imread
import os
import multiprocessing
import sys
import json
import argparse
from caption_generator import *

from inference_utils import extract_features, extract_image_id, run_inference

from voting import rrv_captions_from_beam

import h5py

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None
verbose = True
mode = 'inference'

# Run inference but do beam search in batches for better efficiency...
def run_inference2(sess, features, generator, data, batch_size):
    generator.beam_search2(sess, features, batch_size=batch_size)

def create_annotations(features, image_names, data, num_processes, saved_sess, beam_size=3, voting_scheme="range", num_winners=1, normalise_votes=False):
    # Build the model.
    model = build_model(model_config, mode, inference_batch=1)

    # Initialize beam search Caption Generator
    generator = CaptionGenerator(
        model,
        data['word_to_idx'],
        max_caption_length=model_config.padded_length - 1,
        beam_size=beam_size)
    # run training
    init = tf.global_variables_initializer()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1/(2 * num_processes))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #with tf.Session() as sess:
        sess.run(init)
        model['saver'].restore(sess, saved_sess)

        # predictions
        beam_preds = run_inference(
            sess,
            features,
            generator)

        annotations = []

        for j, beam_captions in enumerate(beam_preds):
            beam_dec = []
            total_prob = 0
            for caption in beam_captions:
                sentence = ' '.join(decode_captions(caption.sentence, data['idx_to_word']))
                prob = np.exp(caption.score)
                beam_dec.append({
                    'caption': sentence,
                    'prob': prob
                })
                total_prob += prob
            print(total_prob)

            voted_captions = rrv_captions_from_beam(beam_captions, num_winners=num_winners, normalise_votes=normalise_votes)

            voted_dec = []
            for voted_caption in voted_captions:
                vote_dec = decode_captions(voted_caption, data['idx_to_word'])
                vote_dec = ' '.join(vote_dec)
                voted_dec.append(vote_dec)

            image_name = image_names[j]

            annotation = {
                'image_id': extract_image_id(image_name),
                'captions': {
                    'beam': beam_dec,
                    'voted': voted_dec
                }
            }
            annotations.append(annotation)

        print("Created annotations for {} images".format(len(features)))
        return annotations


def main(_):
    # Parameters
    voting_scheme = "reweighted"
    num_winners = 3
    beam_size = 100
    batch_size = 2
    normalise_votes = False

    num_processes = 4

    # load dictionary
    data = {}
    with open(FLAGS.dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v
    data['idx_to_word'] = {int(k): v for k, v in data['idx_to_word'].items()}

    print("Loaded dictionary...")
    print("Dictionary size: {}".format(len(data['idx_to_word'])))

    multiprocessing.set_start_method("spawn")

    if FLAGS.load_features:
        features = np.load(os.path.join(FLAGS.test_dir + "features.npy"))
        all_image_names = np.load(os.path.join(FLAGS.test_dir + "image_names.npy"))
        assert(len(features) == len(all_image_names))
        print("Loaded {} features from {}".format(len(features), FLAGS.test_dir))
    else:
        with multiprocessing.Pool(processes=num_processes) as p:
            # extract all features
            features, all_image_names = p.apply(extract_features, args=(FLAGS.test_dir, FLAGS.pretrain_dir))
            tf.reset_default_graph()
            print("Features extracted... Shape: {}".format(features.shape))

        all_image_names = all_image_names['file_name'].values

        np.save(os.path.join(FLAGS.test_dir, "features.npy"), features)
        np.save(os.path.join(FLAGS.test_dir, "image_names.npy"), all_image_names)
        print("Saved features and names to {}".format(FLAGS.test_dir))

    num_of_images = len(features)
    print("Inferencing on {} images".format(num_of_images))

    num_batches = num_of_images // batch_size + (1 if num_of_images %  batch_size != 0 else 0)

    features_batches = [features[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    image_names_batches = [all_image_names[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    print("Number of batches: {}".format(len(features_batches)))

    if FLAGS.run_parallel:
        with multiprocessing.Pool() as p:
            results = [
                p.apply_async(create_annotations, args=(features, image_names), kwds={
                    "data": data,
                    "num_processes": num_processes,
                    "saved_sess": FLAGS.saved_sess,
                    "beam_size": beam_size,
                    "voting_scheme": voting_scheme,
                    "num_winners": num_winners,
                    "normalise_votes": normalise_votes
                })
                for (features, image_names) in zip(features_batches, image_names_batches)]

            annotations = []
            for result in results:
                annotations.extend(result.get())
    else:
        annotations = []
        for (features, image_names) in zip(features_batches, image_names_batches):
            anns = create_annotations(features,
                                image_names,
                                data=data,
                                num_processes=num_processes,
                                saved_sess=FLAGS.saved_sess,
                                beam_size=beam_size,
                                voting_scheme=voting_scheme,
                                num_winners=num_winners,
                                normalise_votes=normalise_votes)

            annotations.extend(anns)

    print("Created {} annotations".format(len(annotations)))
    # Initialise output file
    with open(FLAGS.save_json_file, 'w') as outfile:
        json.dump(annotations, outfile)
    print("Saved output file {}".format(FLAGS.save_json_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrain_dir',
        type=str,
        default='/tmp/imagenet/',
        help="""\
      Path to pretrained model (if not found, will download from web)\
      """
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/home/ubuntu/COCO/testImages/',
        help="""\
      Path to dir of test images to be predicted\
      """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/home/ubuntu/COCO/savedTestImages/',
        help="""\
      Path to dir of predicted test images\
      """
    )
    parser.add_argument(
        '--saved_sess',
        type=str,
        default="/home/ubuntu/COCO/savedSession/model0.ckpt",
        help="""\
      Path to saved session\
      """
    )
    parser.add_argument(
        '--dict_file',
        type=str,
        default='/home/ubuntu/COCO/dataset/COCO_captioning/coco2014_vocab.json',
        help="""\
      Path to dictionary file\
      """
    )
    parser.add_argument(
        '--save_json_file',
        type=str,
        default="../outputs/annotations.json",
        help="Store the resulting annotations in a json file"
    )

    parser.add_argument(
        '--save_output_images',
        type=bool,
        default=False,
        help="Set to True to save annotated images to disk (requires matplotlib)"
    )

    parser.add_argument(
        '--load_features',
        type=bool,
        default=False,
        help="Set to True if the image features (embeddings) should be loaded from disk"
    )

    parser.add_argument(
        '--run_parallel',
        type=bool,
        default=False,
        help="Set to true if multiple process should create captions in parallel"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








