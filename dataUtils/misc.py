
from collections import OrderedDict
import collections
import json

from dataUtils.static_dict import WEIGHT_DICT

class Alphabet:
    """
        Class: Alphabet
        Function: Build vocab
        Params:
              ******    id2words:   type(list),
              ******    word2id:    type(dict)
              ******    vocab_size: vocab size
              ******    min_freq:   vocab minimum freq
              ******    fixed_vocab: fix the vocab after build vocab
              ******    max_cap: max vocab size
    """
    def __init__(self, min_freq=1):
        self.id2words = ['<padding>', 'unseen']
        self.words2id = collections.OrderedDict()
        self.vocab_size = 2
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def build(self, counter):
        """
        :param data:
        :return:
        """
        for key in counter:
            if counter[key] >= self.min_freq:
                self.loadWord2idAndId2Word(key)
        self.fixed_vocab = True

    def loadWord2idAndId2Word(self, string):
        """
        :param string:
        :return:
        """
        if string in self.words2id:
            return self.words2id[string]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(string)
                self.words2id[string] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                return 1 # unseen word's index is 1

def json_iter(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            yield json.loads(line)

def get_vocabulary_and_maxlgth(sentence_samples):
    max_length = 0
    counter = OrderedDict()

    for sample in sentence_samples:
        text = sample['text']
        char_size = len(text)
        for s in text:
            if s not in counter:
                counter[s] = 1
            else:
                counter[s] += 1
        if char_size > max_length:
            max_length = char_size

    char_alphabet = Alphabet(min_freq=3)
    char_alphabet.build(counter)

    return char_alphabet, counter, max_length

def get_bmes_tag(lght):
    if lght == 1:
        tmp = 'S'
    else:
        tmp = ''.join(['B','M' * (lght - 2),'E'])
    return list(tmp)

def postag2char_pos_bmes(postags, word_key = 'word', other_keys = ['pos'], check_length = False, text = None):
    char_pos = [[] for _ in range(len(other_keys))]
    char_bmes = []
    if not check_length:
        for dct in postags:
            lght = len(dct[word_key])
            tmp = get_bmes_tag(lght)
            char_bmes += tmp
            for i, key in enumerate(other_keys):
                char_pos[i] += [dct[key]] * lght
    else:
        assert text is not None

        text_pointer = 0
        for dct in postags:
            word = dct[word_key]
            lght = len(word)
            bmes_tag = get_bmes_tag(lght)
            found = False
            word_pointer = 0
            while True:
                if word_pointer >= lght:
                    break
                if text_pointer >= len(text):
                    break
                if word[word_pointer] == text[text_pointer]:
                    char_bmes += bmes_tag[word_pointer]
                    for i, key in enumerate(other_keys):
                        char_pos[i] += [dct[key]]
                    if word_pointer == lght - 1:
                        found = True
                    word_pointer += 1
                    text_pointer += 1
                else:
                    if word_pointer == 0:
                        char_bmes += ['S']
                        for i, key in enumerate(other_keys):
                            char_pos[i] += [{'postag':'w','netag':'O'}[key]]
                    else:
                        char_bmes += ['M']
                        for i, key in enumerate(other_keys):
                            char_pos[i] += [dct[key]]
                    text_pointer += 1

            if not found:
                raise(Exception(word + ' not found in:\n' + text))

        tail_space = len(text) - len(char_bmes)
        if tail_space > 0:  # 如果空格出现在末尾
            char_bmes += 'S' * tail_space
            for i, key in enumerate(other_keys):
                char_pos[i] += [{'postag': 'w', 'netag': 'O'}[key]] * tail_space
        elif tail_space < 0:
            raise(Exception('impossible!'))
        else:
            pass

    if len(other_keys) == 1:
        char_pos = char_pos[0]
    return char_pos, char_bmes

def my_finditer(lookfor, text):
    lgth = len(lookfor)
    lookfor = lookfor.lower()
    text = text.lower()
    start_from = 0
    while True:
        begin = text.find(lookfor, start_from)
        if begin == -1:
            break
        start_from = begin+lgth
        yield (begin, begin+lgth)

def calculate_weight(label):
    weight_dct = WEIGHT_DICT
    def decide(left, right, now):
        if now == 1:
            return 'T'
        elif (now == 0) and (left + right) > 0:
            return 'E'
        else:
            return 'F'
    left = 0
    res = []
    for i, value in enumerate(label):
        right = value
        if i > 0:
            res.append(decide(left, right, now))
            left = now
        now = value
        if i == len(label) - 1:
            right = 0
            res.append(decide(left, right, now))
    return [weight_dct[k] for k in res]