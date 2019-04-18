from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
import numpy as np
import re

from dataUtils.misc import postag2char_pos_bmes, my_finditer, calculate_weight
from dataUtils.static_dict import POS_VOCA, BMES_VOCA, NER_VOCA, WEIGHT_DICT

def add_to_input(gene):
    def wrapper(*args, **kwargs):
        sample = args[1]
        sample.update(gene(*args, **kwargs))
        return sample
    return wrapper

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
            dct = self.replace_raw_text(dct)
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
    def replace_raw_text(self, sample):
        def mask_books(text):
            parts = [];
            last_tail = 0
            for match in re.finditer('《[^》]*》', text):
                parts.append(text[last_tail:match.span()[0]])
                last_tail = match.span()[1]
                parts.append(''.join(['《', 'X' * (match.span()[1] - match.span()[0] - 2), '》']))
            parts.append(text[last_tail:])
            return ''.join(parts)
        text = sample['text']
        new_text = re.sub('[A-Z]', 'B', text)     # 大写英文字母统一变为B
        new_text = re.sub('[a-z]', 'b', new_text)  # 小写英文字母统一变为b
        new_text = re.sub('[0-9]', 'x', new_text)  # 数字统一变为x
        new_text = mask_books(new_text)  # 中文书名号《》内统一变为X
        res = {'raw_text': text,
               'text': new_text,
               '_id': sample['_id']}
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




