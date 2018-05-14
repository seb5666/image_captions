import os
from tqdm import tqdm
from json import JSONDecodeError
from utils import load_vocab, load_annotations, load_caption, decode_caption, rrv_votes, rrv_votes_hidden_vector

vocab_file = "./outputs/vocab/5000/coco2014_vocab.json"
beam_captions_dir = "./outputs/beam_captions/"
save_file = './outputs/vote_captions.pickle' # File to save generated votes

annotations_dir = "./../annotations/"
annotations_file = "captions_val2014.json"
map_file = "./outputs/val_image_id_to_idx.csv"

image_dir = "./outputs/beam_captions/"

vocab = load_vocab(dict_file = vocab_file)
image_id_to_index, index_to_image_id, annotations_dict = load_annotations(annotations_dir=annotations_dir,
                                                                          annotations_file=annotations_file,
                                                                         map_file = map_file)

print("Found annotatoins for {} images".format(len(image_id_to_index)))
assert(len(image_id_to_index) == len(annotations_dict.keys()))

vote_captions = {}
images = os.listdir(beam_captions_dir)
print("To process: {}".format(len(images)))

for i, image in enumerate(tqdm(images)):
    image_id = int(image.split('.')[0])
    try:
        caption_object = load_caption(image_id, image_dir=image_dir)
        vote_captions[image_id] = rrv_votes_hidden_vector(caption_object, num_winners=5)
    except JSONDecodeError:
        print("Error on ", image_id)

    if i == 2:
        break

with open(save_file, 'wb') as handle:
    pickle.dump(vote_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Evaluate using BLEU
annotation_captions = []
voted_captions = []
highest_beam_probs = []

for image_id in tqdm(vote_captions):
    annotation_captions.append(annotations_dict[image_id])
    voted_captions.append(decode_caption(vote_captions[image_id][0][0], vocab))

beam_bleu_1 = bleu_score.corpus_bleu(annotation_captions, best_beam_caption, weights=[1])
beam_bleu_4 = bleu_score.corpus_bleu(annotation_captions, best_beam_caption, weights=[0.25, 0.25, 0.25, 0.25])

vote_blue_1 = bleu_score.corpus_bleu(annotation_captions, best_voted_caption, weights=[1])
vote_blue_4 = bleu_score.corpus_bleu(annotation_captions, best_voted_caption, weights=[0.25, 0.25, 0.25, 0.25])

print("Beam scores")
print("Bleu-1: {:.4f}".format(beam_bleu_1))
print("Bleu-1: {:.4f}".format(beam_bleu_4))
print()

print("Vote scores")
print("Bleu-1: {:.4f}".format(vote_bleu_1))
print("Bleu-1: {:.4f}".format(vote_bleu_4))
