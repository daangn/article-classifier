from collections import Counter

import pandas as pd

MIN_COUNT = 10

def username_char_dict():
    df = pd.read_csv('data/emb.csv')

    chars_list = df['user_name'].apply(lambda x: list(str(x).decode('utf-8')))
    chars = [char for word in chars_list for char in word]

    print('total chars count', len(set(chars)))

    c = Counter(chars)
    write_char_dict(c.most_common(), 'data/char_dict.txt', MIN_COUNT)

def write_char_dict(most_common, filepath, min_count):
    with open(filepath, 'w') as f:
        for char, count in most_common:
            if count < min_count:
                break
            f.write("%s\n" % char.encode('utf-8'))

def get_chars(filepath):
    chars = []
    with open(filepath, 'r') as f:
        for line in f:
            chars += list(line.decode('utf-8').rstrip().replace(' ', ''))
    return chars

def text_char_dict():
    filepath = 'data/title_normalized.txt.emb.words'
    title_chars = get_chars(filepath)

    filepath = 'data/content_normalized.txt.emb.words'
    content_chars = get_chars(filepath)

    c = Counter(title_chars + content_chars)
    write_char_dict(c.most_common(), 'data/text_char_dict.txt', MIN_COUNT)

if __name__ == '__main__':
    username_char_dict()
    text_char_dict()
