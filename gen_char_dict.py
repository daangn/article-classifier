from collections import Counter

import pandas as pd

MIN_COUNT = 10

df = pd.read_csv('data/emb.csv')

chars_list = df['user_name'].apply(lambda x: list(str(x).decode('utf-8')))
chars = [char for word in chars_list for char in word]

print('total chars count', len(set(chars)))

c = Counter(chars)

with open('data/char_dict.txt', 'w') as f:
    for char, count in c.most_common():
        if count < MIN_COUNT:
            break
        f.write("%s\n" % char.encode('utf-8'))
