from typing import Iterator, Dict
from itertools import chain
from functools import partial
from collections import Counter
import tensorflow as tf


UNK_TOKEN = '<unk>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'


def build_vocab(tokens: Iterator[str]) -> Dict[str, int]:
    counter = Counter(tokens)
    words, _ = zip(*counter.most_common())
    words = (UNK_TOKEN, BOS_TOKEN, EOS_TOKEN) + words
    return dict(zip(words, range(len(words))))


def indexer(vocab, unk_index, tokens):
    return [vocab.get(x, unk_index) for x in [BOS_TOKEN] + tokens + [EOS_TOKEN]]


def load_dataset(src_path: str, tgt_path: str):
    # Loading
    with open(src_path) as f:
        src = [line.split() for line in f]
    with open(tgt_path) as f:
        tgt = [line.split() for line in f]

    # Building vocabulary
    src_vocab = build_vocab(chain.from_iterable(src))
    tgt_vocab = build_vocab(chain.from_iterable(tgt))

    # Indexing
    src_indexer = partial(indexer, src_vocab, src_vocab[UNK_TOKEN])
    tgt_indexer = partial(indexer, tgt_vocab, tgt_vocab[UNK_TOKEN])

    def generator():
        for x, y in zip(src, tgt):
            yield (src_indexer(x), tgt_indexer(y))

    return tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64)), len(src_vocab), len(tgt_vocab)
