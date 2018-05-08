import json
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
sys.path.append("../inference") # go to parent dir
sys.path.append("../utils") # go to parent dir

from voting import rrv_captions
from similarities import unigram_precision, bigram_overlap, bleu_similarity
from prepare_captions import preprocess_json_files

def load_vocab(dict_file):
    data = {}
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            print(k)
            data[k] = v
    data = {int(k): v for k, v in data['idx_to_word'].items()}

    print("Loaded dictionary...")
    print("Dictionary size: {}".format(len(data)))
    return data


def decode_caption(caption, vocab):
    return [vocab[w] for w in caption if w > 3]


def print_image(image_id, image_dir = "../../val2014_2/val2014/"):
    image_path = "{}{}.jpg".format(image_dir, image_id)
    img=mpimg.imread(image_path)
    imgplot = plt.imshow(img)

def load_caption(image_id, image_dir="../outputs/beam_captions/"):
    if type(image_id) == str:
        filename = "{}{}".format(image_dir, image_id)
    else:
        filename = "{}{}.json".format(image_dir, image_id)
    with open(filename, "r") as file:
        return json.load(file)

def load_annotations(annotations_dir="../../annotations/", annotations_file='captions_val2014.json', map_file = "../outputs/val_image_id_to_idx.csv"):
    image_id_to_index, index_to_image_id = _load_image_indexes_map(map_file)

    annotations = preprocess_json_files(annotations_dir)[annotations_file]
    annotations_dict = {}
    for i in range(len(annotations[0])):
        caption = annotations[0][i]
        image_id = annotations[1][i]
        image_index = image_id_to_index[image_id]
        if image_index in annotations_dict:
            annotations_dict[image_index].append(caption)
        else:
            annotations_dict[image_index] = [caption]

    return image_id_to_index, index_to_image_id, annotations_dict

def _load_image_indexes_map(map_file = "../outputs/val_image_id_to_idx.csv"):
    with open(map_file) as file:
        csv_file = csv.reader(file)
        image_id_to_index = {}
        index_to_image_id = {}
        for image_index, image_id in csv_file:
            try:
                image_idx = int(image_index)
                image_id = int(image_id)
                image_id_to_index[image_id] = image_idx
                index_to_image_id[image_idx] = image_id
            except:
                print("Error proccessing {}: {}".format(image_id, image_index))
    return image_id_to_index, index_to_image_id

def rrv_votes(caption_object, num_winners=5, normalise_votes=False, similarity="unigram"):
    if similarity == "unigram_multiplicicy":
        return rrv_captions(caption_object['captions'],
                            caption_object['probabilities'],
                            num_winners=num_winners,
                            normalise_votes=normalise_votes,
                            similarity=unigram_precision)
    elif similarity == "bleu":
        return rrv_captions(caption_object['captions'],
                            caption_object['probabilities'],
                            num_winners=num_winners,
                            normalise_votes=normalise_votes,
                            similarity=bleu_similarity)
    elif similarity == "bigram":
        return rrv_captions(caption_object['captions'],
                            caption_object['probabilities'],
                            num_winners=num_winners,
                            normalise_votes=normalise_votes,
                            similarity=bigram_overlap)
    else:
        return rrv_captions(caption_object['captions'],
                        caption_object['probabilities'],
                        num_winners=num_winners,
                        normalise_votes=normalise_votes)


if __name__== "__main__":
    dict_file = "../outputs/vocab/5000/coco2014_vocab.json"
    vocab = load_vocab(dict_file)