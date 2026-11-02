"""Shared helper functions: phonetics, WordNet synonyms, word maps and the 5 pair types."""
import re

import numpy as np
import cmudict
import config  # sets NLTK_DATA so WordNet loads from data/nltk_data
from nltk.corpus import wordnet


# ---------------- phonetics ----------------

def load_cmudict():
    return cmudict.dict()


def get_phones(word, cmu):
    # first pronunciation only, stress digits removed (AO1 -> AO), as in the paper's code
    if word not in cmu:
        return None
    return [re.sub(r"\d+", "", p) for p in cmu[word][0]]


def edit_distance(a, b):
    # Levenshtein distance between two lists
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def phone_distance(a, b):
    # edit distance normalized by the longer word (same number as jiwer's WER in the paper's code)
    return edit_distance(a, b) / max(len(a), len(b))


# ---------------- semantics ----------------

def get_synonyms(word):
    synonyms = set()
    for synset in wordnet.synsets(word):
        for name in synset.lemma_names():
            if name != word:
                synonyms.add(name)
    return sorted(synonyms)


# ---------------- word maps ----------------

def sample_words(df, seed=None, speaker=None, n=10000):
    # one speaker: all their words; otherwise n random words
    if speaker is not None:
        return df[df.speaker == speaker]
    return df.sample(n, random_state=seed)


def get_word_info(df):
    # word -> (phones, synonyms), one entry per distinct word
    info = {}
    for row in df.itertuples():
        if row.word not in info:
            info[row.word] = (row.phones.split(), row.synonyms.split())
    return info


def build_synonym_map(df, threshold):
    # word -> synonyms present in the sample; threshold=None keeps all of them,
    # otherwise only synonyms that sound different (distance > threshold)
    info = get_word_info(df)
    synonym_map = {}
    for word, (phones, synonyms) in info.items():
        found = set()
        for s in synonyms:
            if s not in info:
                continue
            if threshold is None or phone_distance(phones, info[s][0]) > threshold:
                found.add(s)
        if len(found) > 0:
            synonym_map[word] = found
    return synonym_map


def build_homophone_map(df, all_synonyms, threshold):
    # word -> sound-alike words (0 < distance <= threshold) that are not synonyms
    info = get_word_info(df)
    words = list(info.keys())
    homophone_map = {}
    for word in words:
        phones = info[word][0]
        found = set()
        for other in words:
            if other == word:
                continue
            if other in all_synonyms.get(word, set()) or word in all_synonyms.get(other, set()):
                continue
            other_phones = info[other][0]
            # cheap skip: a big length difference alone already exceeds the threshold
            longer = max(len(phones), len(other_phones))
            if abs(len(phones) - len(other_phones)) / longer > threshold:
                continue
            d = phone_distance(phones, other_phones)
            if 0 < d <= threshold:
                found.add(other)
        if len(found) > 0:
            homophone_map[word] = found
    return homophone_map


# ---------------- the 5 pair types ----------------
# Each returns a list of (left_row, right_row) using the word table's row ids.
# Shuffles use a fixed seed of 42, like the paper's code.

def shuffled_pairs(rows):
    rows = np.array(rows)
    shuffled = rows.copy()
    np.random.default_rng(seed=42).shuffle(shuffled)
    return [(int(l), int(r)) for l, r in zip(rows, shuffled) if l != r]


def rows_by_word(df):
    groups = {}
    for row_id, word in zip(df.index, df.word):
        groups.setdefault(word, []).append(row_id)
    return groups


def random_pairs(df):
    return shuffled_pairs(df.index)


def same_speaker_pairs(df):
    pairs = []
    for speaker in df.speaker.unique():
        pairs += shuffled_pairs(df[df.speaker == speaker].index)
    return pairs


def same_word_pairs(df):
    pairs = []
    for rows in rows_by_word(df).values():
        pairs += shuffled_pairs(rows)
    return pairs


def map_pairs(df, word_map):
    # every token paired with every token of each word it maps to
    groups = rows_by_word(df)
    pairs = []
    for left, word in zip(df.index, df.word):
        for other in sorted(word_map.get(word, set())):
            for right in groups[other]:
                pairs.append((int(left), int(right)))
    return pairs


def synonym_pairs(df, synonym_map):
    return map_pairs(df, synonym_map)


def homophone_pairs(df, homophone_map):
    return map_pairs(df, homophone_map)
