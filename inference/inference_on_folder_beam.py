
"""Predict captions on test images using trained model, with beam search method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
from ShowAndTellModel import build_model
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
import numpy as np
import os
import sys
import json
import argparse

from inference_utils import extract_features, extract_image_id, run_inference

from caption_generator import * 

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None
verbose = True
mode = 'inference'

def save_beam_captions(features, image_names, data, saved_sess, beam_size=3, batch_size = 16):
    model = build_model(model_config, mode, inference_batch=1)

    generator = CaptionGenerator(
        model,
        data['word_to_idx'],
        max_caption_length=model_config.padded_length - 1,
        beam_size=beam_size
    )

    num_batches = len(features) // batch_size + (1 if len(features) % batch_size != 0 else 0)

    features_batches = [features[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]
    image_names_batches = [image_names[i * batch_size : (i+1) * batch_size] for i in range(num_batches)]

    init = tf.global_variables_initializer()

    total_images_processed = 0

    with tf.Session() as sess:
        sess.run(init)
        model['saver'].restore(sess, saved_sess)

        for j, (features_batch, image_names_batch) in enumerate(zip(features_batches, image_names_batches)):

            features = []
            image_names = []

            for i, image_name in enumerate(image_names_batch):
                image_id = extract_image_id(image_name)

                if FLAGS.starts_with is not None and not str(image_id).startswith(FLAGS.starts_with):
                    continue
                output_file = os.path.join(FLAGS.save_dir, str(image_id) + ".json")
                if not os.path.isfile(output_file):
                    features.append(features_batch[i])
                    image_names.append(image_name)

            if len(features) == 0:
                print("Skipping batch {}".format(j + 1))
                continue

            total_images_processed += len(features)

            features = np.array(features)

            beam_predictions_batch = run_inference(sess, features, generator)

            for beam_predictions, image_name in zip(beam_predictions_batch, image_names):
                image_id = extract_image_id(image_name)
                output_file = os.path.join(FLAGS.save_dir, str(image_id) + ".json")

                total_prob = 0
                scores = []
                captions = []
                for caption in beam_predictions:
                    score = np.exp(caption.score)
                    captions.append(caption.sentence)
                    scores.append(score)
                    total_prob += score

                beam_captions = {
                    'image_id': image_id,
                    'captions': captions,
                    'probabilities': scores,
                    'total_prob': total_prob
                }

                print("Saving to {}".format(output_file))
                # Initialise output file
                with open(output_file, 'w+') as outfile:
                    json.dump(beam_captions, outfile)

            print("Saved beams for batch {} out {}".format(j + 1, num_batches))

        print("Created annotations for {} images".format(total_images_processed))


def main(_):

    # Parameters
    beam_size = 100

    # load dictionary
    data = {}
    with open(FLAGS.dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v
    data['idx_to_word'] = {int(k): v for k, v in data['idx_to_word'].items()}

    print("Loaded dictionary...")
    print("Dictionary size: {}".format(len(data['idx_to_word'])))


    if FLAGS.load_features:
        features = np.load(os.path.join(FLAGS.test_dir + "features.npy"))
        all_image_names = np.load(os.path.join(FLAGS.test_dir + "image_names.npy"))
        assert(len(features) == len(all_image_names))
        print("Loaded {} features from {}".format(len(features), FLAGS.test_dir))
    else:
        features, all_image_names = extract_features(FLAGS.test_dir, FLAGS.pretrain_dir)
        print("Features extracted... Shape: {}".format(features.shape))
        all_image_names = all_image_names['file_name'].values

        np.save(os.path.join(FLAGS.test_dir, "features.npy"), features)
        np.save(os.path.join(FLAGS.test_dir, "image_names.npy"), all_image_names)
        print("Saved features and names to {}".format(FLAGS.test_dir))

    num_of_images = len(features)
    print("Inferencing on {} images".format(num_of_images))
    save_beam_captions(features, all_image_names, data=data, saved_sess=FLAGS.saved_sess, beam_size=beam_size)
    print("Done")

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
        '--save_dir',
        type=str,
        default="../outputs/beam_captions/",
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

    parser.add_argument(
        '--starts_with',
        type=str,
        default=None,
        help="Only consider images whose id starts with the given value. None to consider all"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





