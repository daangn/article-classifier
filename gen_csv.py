import sys
import logging
from os import listdir
from os.path import isfile, join
from random import shuffle
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from trainer.emb import LABEL_COL

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in xrange(0, len(l), n):
		yield l[i:i + n]

def files(path):
  return [f for f in listdir(path) if isfile(join(path, f))]

def dirs(path):
  return [f for f in listdir(path) if not isfile(join(path, f))]

EVAL_RATIO = 0.25
CHUNK_SIZE = 10000

def main():
    with open('data/emb.csv') as f:
        X = f.readlines()[1:]
    y = [x.split(',')[LABEL_COL].rstrip() for x in X]

    X = np.array(X)
    y = np.array(y)

    print(X)
    print(y)

    print('All y counter')
    print(Counter(y).most_common())

    sss = StratifiedShuffleSplit(n_splits=1, test_size=EVAL_RATIO)
    train_index, test_index = next(sss.split(X, y))

    print('total count: %d' % X.shape[0])
    K = np.zeros((X.shape[0]), np.int32)
    K[test_index] = 1

    import math, random
    train_set_size = int(math.ceil(1.0 * len(train_index) / CHUNK_SIZE))
    eval_set_size = int(math.ceil(1.0 * len(test_index) / CHUNK_SIZE))

    def set_file_open(name, i):
        return open("data/%s_set%d.csv" % (name, i), 'w')

    files = [
        [set_file_open('train', i) for i in range(train_set_size)],
        [set_file_open('eval', i) for i in range(eval_set_size)],
    ]

    with open('data/text_normalized.txt.emb') as f:
        for i, line in enumerate(f):
            kind = K[i]
            random.choice(files[kind]).write("%s,%s" % (X[i].rstrip(), line))

    for kind_files in files:
        for f in kind_files:
            f.close()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main()
