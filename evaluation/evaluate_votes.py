import os
from tqdm import tqdm
from json import JSONDecodeError
from utils import load_vocab, load_annotations, load_caption, decode_caption, rrv_votes, git_hidden_vector
import pickle
from nltk.translate import bleu_score

LOAD_CAPTIONS = False

beam_size = 10
use_hidden = False
vocab_file = "/home/spb61/coco2014_vocab.json"
beam_captions_dir = "/datadrive/val_beam_{}_states/".format(beam_size)
print("Loading beams from: {}".format(beam_captions_dir))
save_file = "/home/spb61/image_captions/outputs/vote_captions_scores_{}_bigram_precision.pickle".format(beam_size) # File to save generated votes

annotations_dir = "/home/spb61/annotations/"
annotations_file = "captions_val2014.json"
map_file = "/home/spb61/val_image_id_to_idx.csv"

vocab = load_vocab(dict_file = vocab_file)
image_id_to_index, index_to_image_id, annotations_dict = load_annotations(annotations_dir=annotations_dir,
                                                                          annotations_file=annotations_file,
                                                                         map_file = map_file)

print("Found annotatoins for {} images".format(len(image_id_to_index)))
assert(len(image_id_to_index) == len(annotations_dict.keys()))

if not LOAD_CAPTIONS:
    print("Saving voted caps to: {}".format(save_file))
    vote_captions = {}
    images = os.listdir(beam_captions_dir)
    print("To process: {}".format(len(images)))

    for i, image in enumerate(tqdm(images)):
        image_id = int(image.split('.')[0])
        try:
            caption_object = load_caption(image_id, image_dir=beam_captions_dir)
            if use_hidden:
                vote_captions[image_id] = rrv_votes_hidden_vector(caption_object['captions'], num_winners=1)
            else:
                caption = {}
                caption['probabilities'] = [c['score'] for c in caption_object['captions']]
                caption['captions'] = [c['sentence'] for c in caption_object['captions']]
                vote_captions[image_id] = rrv_votes(caption, num_winners=1, similarity='bigram_overlap')
        except JSONDecodeError:
            print("Error on ", image_id)

    with open(save_file, 'wb') as handle:
        pickle.dump(vote_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print("Loading captions from {}".format(save_file))
    with open(save_file, 'rb') as handle:
        vote_captions = pickle.load(handle)


# Evaluate using BLEU
annotation_captions = []
voted_captions = []
highest_beam_probs = []
best_beam_caption = []

for image_id in tqdm(vote_captions):
    annotation_captions.append(annotations_dict[image_id])
    voted_captions.append(decode_caption(vote_captions[image_id][0][0], vocab))
    loaded_cap = load_caption(image_id, image_dir=beam_captions_dir)
    best_beam_caption.append(decode_caption(loaded_cap['captions'][0]['sentence'], vocab))

beam_bleu_1 = bleu_score.corpus_bleu(annotation_captions, best_beam_caption, weights=[1])
beam_bleu_4 = bleu_score.corpus_bleu(annotation_captions, best_beam_caption, weights=[0.25, 0.25, 0.25, 0.25])

vote_bleu_1 = bleu_score.corpus_bleu(annotation_captions, voted_captions, weights=[1])
vote_bleu_4 = bleu_score.corpus_bleu(annotation_captions, voted_captions, weights=[0.25, 0.25, 0.25, 0.25])

print("Beam scores")
print("Bleu-1: {:.4f}".format(beam_bleu_1))
print("Bleu-4: {:.4f}".format(beam_bleu_4))
print()

print("Vote scores")
print("Bleu-1: {:.4f}\t{}".format(vote_bleu_1, vote_bleu_1))
print("Bleu-4: {:.4f}\t{}".format(vote_bleu_4, vote_bleu_4))
