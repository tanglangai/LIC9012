from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from collections import OrderedDict
import collections
import numpy as np
import json

from dataloader.alphabet import Alphabet
from dataloader.static_dict import POS_VOCA, BMES_VOCA, NER_VOCA, WEIGHT_DICT

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

def add_to_input(gene):
    def wrapper(*args, **kwargs):
        sample = args[1]
        sample.update(gene(*args, **kwargs))
        return sample
    return wrapper

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

class Char_Feature_Pipeline():
    def __init__(self, char_voca, freq_dict, pyltp_path):
        self.char_voca = char_voca
        self.freq_dict = freq_dict
        self.pyltp_path = pyltp_path
        PYLTP_PATH = self.pyltp_path
        self.segmentor = Segmentor()
        self.segmentor.load(PYLTP_PATH + '/cws.model')
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(PYLTP_PATH + '/pos.model')  # 加载模型
        self.recognizer = NamedEntityRecognizer()  # 初始化实例
        self.recognizer.load(PYLTP_PATH + '/ner.model')  # 加载模型

    def pipeline(self, raw_iter, label_bool=False):
        for Id, sample in enumerate(raw_iter):
            dct = self.gene_pyltp_feature(sample, Id)  # use pyltp to get new seg, pos and ner
            dct = self.gene_ner_feature(dct)  # generate the NER feature
            if label_bool:
                dct = self.gene_ner_label(dct)  # gene_ner_label
                dct = self.gene_ner_weight(dct)  # gene_label_weight
            yield dct

    def release_pyltp_model(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()

    @add_to_input
    def gene_pyltp_feature(self, sample, Id):
        words = self.segmentor.segment(sample['text'])
        postags = self.postagger.postag(words)
        netags = self.recognizer.recognize(words, postags)
        res = [{'word': word, 'postag': postag, 'netag': netag} for word, postag, netag in
               zip(words, postags, netags)]
        res = {'_id': sample.get('_id', Id), 'pyltp_tags': res}
        return res

    @add_to_input
    def gene_ner_feature(self, sample):
        char_index = [self.char_voca.loadWord2idAndId2Word(char) for char in sample['text']]
        char_pos, char_bmes = postag2char_pos_bmes(sample['postag'])

        ltp_other, ltp_char_bmes = postag2char_pos_bmes(sample['pyltp_tags'],
                                                        word_key='word',
                                                        other_keys=['postag', 'netag'],
                                                        check_length=True,
                                                        text=sample['text'])

        pos_index = [POS_VOCA[pos] for pos in char_pos]
        bmes_index = [BMES_VOCA[k] for k in char_bmes]
        char_freq = [self.freq_dict[s] for s in sample['text']]

        ltp_bmes_index = [BMES_VOCA[k] for k in ltp_char_bmes]
        ltp_pos_index = [POS_VOCA[pos] for pos in ltp_other[0]]
        ltp_ner_index = [NER_VOCA[k] for k in ltp_other[1]]

        if not len(char_pos) == len(char_index):
            pos_index = ltp_pos_index
            bmes_index = ltp_bmes_index

        assert len(char_index) == len(pos_index)

        res = {'_id': sample['_id'],
               'char_index': char_index,
               'char_size': len(sample['text']),
               'pos_index': pos_index,
               'bmes_index': bmes_index,
               'char_freq': char_freq,
               'ltp_pos_index': ltp_pos_index,
               'ltp_bmes_index': ltp_bmes_index,
               'ltp_ner_index': ltp_ner_index}
        return res

    @add_to_input
    def gene_ner_label(self, sample):
        text = sample['text']
        char_length = len(text)
        subjects = set([spo['subject'] for spo in sample['spo_list']])
        objects = set([spo['object'] for spo in sample['spo_list']])
        locates = np.zeros(char_length, dtype=int)
        for bject in subjects:
            for span in my_finditer(bject, text):
                locates[span[0]:span[1]] = 1
        sub_locates = locates.tolist()
        locates = np.zeros(char_length, dtype=int)
        for bject in objects:
            for span in my_finditer(bject, text):
                locates[span[0]:span[1]] = 1
        ob_locates = locates.tolist()
        res = {'_id': sample['_id'],
               'sub_label': sub_locates,
               'ob_label': ob_locates}
        return res

    @add_to_input
    def gene_ner_weight(self, sample):
        sub_weight = calculate_weight(sample['sub_label'])
        ob_weight = calculate_weight(sample['ob_label'])
        res = {'_id': sample['_id'],
               'sub_weight': sub_weight,
               'ob_weight': ob_weight}
        return res


BMES_VOCA = {'B': 0,
             'M': 1,
             'E': 2,
             'S': 3}

POS_VOCA = {'r': 0,
             'v': 1,
             'a': 2,
             'u': 3,
             'n': 4,
             'w': 5,
             'nw': 6,
             'nr': 7,
             'p': 8,
             'f': 9,
             'nz': 10,
             'm': 11,
             'ns': 12,
             'd': 13,
             't': 14,
             'ad': 15,
             'vn': 16,
             's': 17,
             'nt': 18,
             'c': 19,
             'vd': 20,
             'xc': 21,
             'an': 22,
             'q': 23,
             'wp': 24,
             'nh': 25,
             'nd': 26,
             'b': 27,
             'ws': 28,
             'nl': 29,
             'i': 30,
             'j': 31,
             'ni': 32,
             'z': 33,
             'k': 34,
             'o': 35,
             'h': 36,
             'e': 37,
             '%': 38}

NER_VOCA = {'O': 0,
            'S-Nh': 1,
            'S-Ns': 2,
            'B-Ns': 3,
            'E-Ns': 4,
            'B-Ni': 5,
            'I-Ni': 6,
            'E-Ni': 7,
            'B-Nh': 8,
            'E-Nh': 9,
            'I-Ns': 10,
            'S-Ni': 11,
            'I-Nh': 12}

WEIGHT_DICT = {'T': 1., 'E': 2., 'F': 0.2}
