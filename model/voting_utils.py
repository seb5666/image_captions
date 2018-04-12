import numpy as np

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
                yield x
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

