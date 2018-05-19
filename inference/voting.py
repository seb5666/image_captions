import numpy as np

from similarities import unigram_overlap

def range_vote_caption(beam_predictions,  normalise_votes=False):
    sentences = [pred.sentence for pred in beam_predictions]
    scores = [pred.score for pred in beam_predictions]
    scores = np.exp(np.array(scores))

    # Compute pair-wise similarity
    similarity = np.array([[unigram_overlap(p, q) for q in sentences] for p in sentences])

    if normalise_votes:
        similarity = similarity / np.max(similarity, axis=1)[:, np.newaxis]

    vote_pred = beam_predictions[range_vote(similarity, scores)].sentence
    return [np.array(vote_pred)]

def rrv_captions_from_beam(beam_predictions,  num_winners=1, normalise_votes=False):
    """
    Reweighted range vote caption for the beam predictions given
    :param beam_predictions: beam captions
    :param num_winners: number of captions to return
    :param normalise_votes: true if the votes are normalised to the range [0,1]
    :return: A list containing top num_winners captions in the RRV election
    """
    sentences = [pred.sentence for pred in beam_predictions]
    scores = [pred.score for pred in beam_predictions]
    scores = np.exp(np.array(scores))

    return rrv_captions(sentences, scores, num_winners=num_winners, normalise_votes=normalise_votes)

def rrv_captions(sentences, scores, num_winners=1, normalise_votes=False, similarity=unigram_overlap):
    assert (num_winners <= len(scores))

    # Compute pair-wise similarity
    similarity = np.array([[similarity(p, q) for q in sentences] for p in sentences])

    if normalise_votes:
        similarity = similarity / np.max(similarity, axis=1)[:, np.newaxis]

    winners = [winner for (_, winner) in zip(range(num_winners), reweighted_range_vote(similarity, scores))]

    return [np.array(sentences[x[0]]) for x in winners], [scores[x[0]] for x in winners], [x[1] for x in winners]


def range_vote(votes, weights):
    """
    :param votes: N x N numpy array where votes[s][t] indicates the vote sentence s gives to sentence t. Each row sums to 1
    :param weights: a numpy array of size N where weight[s] indicates by how much the votes of s need to be weighted
    :return: The winner of the range voting election
    """

    # compute weighted similarity
    weighted_sim = weights @ votes
    return np.argmax(weighted_sim)

def reweighted_range_vote(votes, weights, max=1):
    """
    :param votes: N x N numpy array where votes[s][t] indicates the vote sentence s gives to sentence t. Each row sums to 1
    :param weights: a numpy array of size N where weight[s] indicates by how much the votes of s need to be weighted
    :return: An iterator giving the winners in order of the RRV
    """

    winners = np.zeros(len(weights))
    discounts = np.ones(len(weights))

    num_winners = 0

    sums = np.zeros(len(weights))

    while num_winners < len(weights):
        scores = (weights * discounts) @ votes
        for x in reversed(np.argsort(scores)):
            if winners[x]:
                continue
            else:
                winners[x] = 1
                sums += votes[:, x]
                discounts = 1 / (1 + sums/max)
                num_winners += 1
                yield x, scores[x]
                break


if __name__ == "__main__":
    votes = np.array(
            [[1,9,3],
             [5,1,1],
             [2,2,7]])

    weights = np.array([1, 1, 1])


    for x in(reweighted_range_vote(votes, weights, max=9)):
        print(x)
        print()

