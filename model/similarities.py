def remove_tags(s):
    """
    Removes tags that should be ignored when computing similarites
    :param s: Sentence as a list of words
    :return: Same sentence without the tags
    """
    tags = set(['<START>', '<END>', '<UNK>'])
    filtered_words = []
    for word in s:
        if word not in tags:
            filtered_words.append(word)
    return filtered_words

def unigram_overlap(s, t):
    """
    Computes the unigram overlap between s and t.
    Note: this is not necessarily symmetric as s and t might have different lengths.
    Note: ignores multiplicity of words
    :param s: sentence 1 as a list of words
    :param t: sentence 2 as a list of words
    :return: The unigram precision
    """

    s = remove_tags(s)
    t = remove_tags(t)

    if len(s) == 0 or len(t) == 0:
        return 0

    # Don't take into account multiplicity
    unique_words = set(s)
    t = set(t)

    overlap = 0
    for w in unique_words:
        overlap += 1 if w in t else 0

    return overlap / len(unique_words)

def bigram_overlap(s, t):
    s = remove_tags(s)
    t = remove_tags(t)
    if len(s) == 0 or len(t) == 0:
        return 0

    bigrams_s = list(zip(s[:,-1], s[1:]))
    bigram_t = list(zip(t[:,-1], t[1:]))

    overlap = 0
    for bigram in bigrams_s:
        overlap += 1 if bigram in bigram_t else 0
    return overlap / len(bigrams_s)