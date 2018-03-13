"""Predict captions on test images using trained model, with beam search method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
from ShowAndTellModel import build_model
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url, write_text_on_image
import numpy as np
from scipy.misc import imread
import os
import sys
import json
import argparse
from caption_generator import *

from inference_utils import extract_features, extract_image_id

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None
verbose = True
mode = 'inference'


def bow_overlap(s, t):
    if len(s) == 0 or len(t) == 0:
        return 0

    if s[0] == '<START>':
        s = s[1:]
    if t[0] == '<START>':
        t = t[1:]
    if s[-1] == '<END>':
        s = s[:-1]
    if t[-1] == '<END>':
        t = t[:-1]

    unique_words = set(s)
    overlap = 0
    for w in unique_words:
        overlap += 1 if w in t else 0
    sim = overlap / len(unique_words)

    return sim

def run_inference(sess, features, generator, keep_prob, data):
    batch_size = features.shape[0]

    vote_preds = []
    beam_preds = []

    for i in range(batch_size):
        print("Batch {}/{}".format(i, batch_size))
        feature = features[i].reshape(1, -1)
        preds = generator.beam_search(sess, feature)

        scores = []
        decoded_preds = []
        for pred in preds:
            score = pred.score
            sentence = decode_captions(np.array(pred.sentence).reshape(-1, 1), data['idx_to_word'])
            scores.append(score)
            decoded_preds.append(sentence)

        scores = np.exp(np.array(scores))

        # Compute pair-wise similarity
        similarity = np.array([[bow_overlap(p, q) for q in decoded_preds] for p in decoded_preds])

        # compute weighted similarity
        weighted_sim = scores @ similarity

        vote_pred = preds[np.argmax(weighted_sim)].sentence
        vote_preds.append(np.array(vote_pred))

        beam_pred = preds[0].sentence
        beam_preds.append(np.array(beam_pred))

    return beam_preds, vote_preds


def main(_):
    # load dictionary
    data = {}
    with open(FLAGS.dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v
    data['idx_to_word'] = {int(k): v for k, v in data['idx_to_word'].items()}

    print("Loaded dictionary...")

    # extract all features
    features, all_image_names = extract_features(FLAGS.test_dir, FLAGS.pretrain_dir)
    print("Features extracted...")

    # Build the TensorFlow graph and train it
    g = tf.Graph()
    with g.as_default():
        num_of_images = len(os.listdir(FLAGS.test_dir))
        print("Inferencing on {} images".format(num_of_images))

        # Build the model.
        model = build_model(model_config, mode, inference_batch=1)

        # Initialize beam search Caption Generator
        generator = CaptionGenerator(model, data['word_to_idx'], max_caption_length=model_config.padded_length - 1, beam_size=10)

        # run training
        init = tf.global_variables_initializer()

        annotations = []

        with tf.Session() as sess:

            sess.run(init)

            model['saver'].restore(sess, FLAGS.saved_sess)

            print("Model restored! Last step run: ", sess.run(model['global_step']))

            # predictions
            beam_preds, vote_preds = run_inference(sess, features, generator, 1.0, data)
            # captions_pred = np.concatenate(captions_pred, 1)
            captions_deco = []
            for beam_pred, vote_pred in zip(beam_preds, vote_preds):
                beam_dec = decode_captions(beam_pred.reshape(-1, 1), data['idx_to_word'])
                beam_dec = ' '.join(beam_dec)

                vote_dec = decode_captions(vote_pred.reshape(-1, 1), data['idx_to_word'])
                vote_dec = ' '.join(vote_dec)
                print("Beam caption:")
                print(beam_dec)
                print("Voted caption:")
                print(vote_dec)
                print()

                captions_deco.append(beam_dec + '\n' + vote_dec)


            # saved the images with captions written on them
            if not os.path.exists(FLAGS.results_dir):
                os.makedirs(FLAGS.results_dir)
            for j in range(len(captions_deco)):
                this_image_name = all_image_names['file_name'].values[j]
                img_name = os.path.join(FLAGS.results_dir, this_image_name)
                img = imread(os.path.join(FLAGS.test_dir, this_image_name))
                write_text_on_image(img, img_name, captions_deco[j])

                annotation = {
                    'image_id': extract_image_id(this_image_name),
                    'caption': captions_deco[j]
                }
                annotations.append(annotation)

        with open(FLAGS.save_json_file, 'w') as outfile:
            json.dump(annotations, outfile)

    print("\ndone.")


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
        default="./annotations.json",
        help="Store the resulting annotiations in a json file"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








