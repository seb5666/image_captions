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

from voting_utils import reweighted_range_vote, range_vote

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

    # Don't take into account multiplicity
    unique_words = set(s)
    overlap = 0
    for w in unique_words:
        overlap += 1 if w in t else 0

    if len(unique_words):
        sim = overlap / len(unique_words)
    else:
        sim = 0

    return sim

# Run inference but do beam search in batches for better efficiency...
def run_inference2(sess, features, generator, data, batch_size):
    generator.beam_search2(sess, features, batch_size=batch_size)

# TODO: make this work for larger batch sizes... Otherwise evaluation will take too long
def run_inference(sess, features, generator, data, num_winners=1, voting_scheme="range"):
    vote_preds = []
    beam_preds = []

    for i in range(len(features)):
        if i % 1000 == 0:
            print("Image {}/{}".format(i, len(features)))
        feature = features[i].reshape(1, -1)

        preds = generator.beam_search(sess, feature)

        scores = [pred.score for pred in preds]
        scores = np.exp(np.array(scores))

        sentences = [pred.sentence for pred in preds]

        # Compute pair-wise similarity
        similarity = np.array([[bow_overlap(p, q) for q in sentences] for p in sentences])

        if voting_scheme == "range":
            vote_pred = preds[range_vote(similarity, scores)].sentence
            vote_preds.append([np.array(vote_pred)])

        elif voting_scheme == "reweighted":
            assert (num_winners <= len(scores))
            winners = [winner for (i, winner) in zip(range(num_winners), reweighted_range_vote(similarity, scores))]
            vote_preds.append([np.array(preds[x].sentence) for x in winners])

        else:
            raise ValueError("Invalid voting scheme {}".format(voting_scheme))

        beam_pred = preds[0].sentence
        beam_preds.append(np.array(beam_pred))

    return beam_preds, vote_preds

def main(_):

    # Parameters
    voting_scheme = "reweighted"
    num_winners = 4
    beam_size = 10

    inference_batch_size = 1


    # load dictionary
    data = {}
    with open(FLAGS.dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v
    data['idx_to_word'] = {int(k): v for k, v in data['idx_to_word'].items()}

    print("Loaded dictionary...")
    print("Dictionary size: {}".format(len(data['idx_to_word'])))

    # extract all features
    features, all_image_names = extract_features(FLAGS.test_dir, FLAGS.pretrain_dir)
    print("Features extracted... Shape: {}".format(features.shape))

    # Build the TensorFlow graph
    g = tf.Graph()
    with g.as_default():
        num_of_images = len(os.listdir(FLAGS.test_dir))
        print("Inferencing on {} images".format(num_of_images))

        # Build the model.
        model = build_model(model_config, mode, inference_batch=inference_batch_size)

        # Initialize beam search Caption Generator
        generator = CaptionGenerator(
            model,
            data['word_to_idx'],
            max_caption_length=model_config.padded_length - 1,
            beam_size=beam_size)

        # run training
        init = tf.global_variables_initializer()

        annotations = []

        with tf.Session() as sess:

            sess.run(init)

            model['saver'].restore(sess, FLAGS.saved_sess)

            print("Model restored! Last step run: ", sess.run(model['global_step']))

            # predictions
            beam_preds, vote_preds = run_inference(sess, features, generator, data, voting_scheme=voting_scheme, num_winners=num_winners)
            # beam_preds, vote_preds = run_inference2(sess, features, generator, data, inference_batch_size)
            # exit()

            for i, (beam_caption, voted_captions) in enumerate(zip(beam_preds, vote_preds)):

                beam_dec = decode_captions(beam_caption, data['idx_to_word'])
                beam_dec = ' '.join(beam_dec)

                voted_dec = []
                for voted_caption in voted_captions:
                    vote_dec = decode_captions(voted_caption, data['idx_to_word'])
                    vote_dec = ' '.join(vote_dec)
                    voted_dec.append(vote_dec)

                image_name = all_image_names['file_name'].values[i]

                annotation = {
                    'image_id': extract_image_id(image_name),
                    'captions': {
                        'beam': beam_dec,
                        'voted': voted_dec
                    }
                }
                annotations.append(annotation)

                if FLAGS.save_output_images:
                    # saved the images with captions written on them
                    if not os.path.exists(FLAGS.results_dir):
                        os.makedirs(FLAGS.results_dir)
                    image_caption = beam_dec + '\n' + '\n'.join(voted_dec)
                    output_image_path = os.path.join(FLAGS.results_dir, image_name)
                    input_image_path = imread(os.path.join(FLAGS.test_dir, image_name))
                    write_text_on_image(input_image_path, output_image_path, image_caption)

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
        help="Store the resulting annotations in a json file"
    )

    parser.add_argument(
        '--save_output_images',
        type=bool,
        default=False,
        help="Set to True to save annotated images to disk (requires matplotlib)"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)








