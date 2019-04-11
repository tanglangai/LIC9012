"""
梁家熙，这里是对common下面的utils进行测试
"""

from common_utils.preprocess_merge_something import *
from unittest import TestCase



class test_merge_adjoint_postag(TestCase):
    
    def testa(self):
        words = ['a', 'b', 'c']
        postags = ['n', 'n', 'r']
        a, b = merge_adjoint_postag(words, postags)
       
        self.assertListEqual(a, ['a', 'b', 'c'])
        self.assertListEqual(b, ['n', 'n', 'r'])

    def testb(self):
        words = ['a', 'b', 'c', 'r', 'f', 'e', 'p']
        postags = ['n', 'r', 'r', 'm', 'm', 'e', 'u']
        a, b = merge_adjoint_postag(words, postags)
    
        self.assertListEqual(a, ['a', 'b', 'c', 'r', 'f', 'e', 'p'])
        self.assertListEqual(b, ['n', 'r', 'r', 'm', 'm', 'e', 'u'])
    def testc(self):
        words = ['a', 'b', 'c']
        postags = ['t', 't', 'r']
        a, b = merge_adjoint_postag(words, postags)
    
        self.assertListEqual(a, ['ab', 'c'])
        self.assertListEqual(b, ['t', 'r'])
    
    
    
    
    
class test_merge_name_punctuation(TestCase):
    def testc(self):
        words = ['爱德华', '·', '彼得', '·', '呵呵', '是']
        postags = ['nr', 'w', 'nr', 'w', 'nr', 't']
        a, b = merge_name_punctuation(words, postags)
        
        self.assertListEqual(a, ['爱德华·彼得·呵呵', '是'])
        self.assertListEqual(b, ['nr', 't'])
    
    def testd(self):
        words = ['fff', '爱德华', '·', '彼得', '·', '呵呵', '是']
        postags = ['a', 'nr', 'w', 'nr', 'w', 'nr', 't']
        a, b = merge_name_punctuation(words, postags)
        
        self.assertListEqual(a, ['fff', '爱德华·彼得·呵呵', '是'])
        self.assertListEqual(b, ['a', 'nr', 't'])
        
        
        
        
        
class test_merge_quotation_marks(TestCase):
    def testa(self):
        words = ['《', '游戏', '人生', '》', '是', '好看', '的']
        postags = ['w', 'n', 'n', 'w', 'f', 'nr', 't']

        a, b = merge_quotation_marks(words, postags)
        self.assertListEqual(a, ['《', '游戏人生', '》', '是', '好看', '的'])
        self.assertListEqual(b, ['w', 'n', 'w', 'f', 'nr', 't'])
    def testb(self):
        words = ['咕咕','《', '游戏', '人生', '》', '是', '好看', '的']
        postags = ['m','w', 'n', 'n', 'w', 'f', 'nr', 't']

        a, b = merge_quotation_marks(words, postags)
        self.assertListEqual(a, ['咕咕','《', '游戏人生', '》', '是', '好看', '的'])
        self.assertListEqual(b, ['m','w', 'n', 'w', 'f', 'nr', 't'])
    def testc(self):
        words = ['咕咕','《', '游戏', '人生', '》']
        postags = ['m','w', 'n', 'n', 'w']

        a, b = merge_quotation_marks(words, postags)
        self.assertListEqual(a, ['咕咕','《', '游戏人生', '》'])
        self.assertListEqual(b, ['m','w', 'n', 'w'])
    def testd(self):
        words = ['咕咕','“', '游戏', '人生', '”', '是', '好看', '的']
        postags = ['m','w', 'n', 'n', 'w', 'f', 'nr', 't']

        a, b = merge_quotation_marks(words, postags)
        self.assertListEqual(a, ['咕咕','“', '游戏人生', '”', '是', '好看', '的'])
        self.assertListEqual(b, ['m','w', 'n', 'w', 'f', 'nr', 't'])
    def teste(self):
        words = ['咕咕','‘', '游戏', '人生', '’', '是', '好看', '的']
        postags = ['m','w', 'n', 'n', 'w', 'f', 'nr', 't']

        a, b = merge_quotation_marks(words, postags)
        self.assertListEqual(a, ['咕咕','‘', '游戏人生', '’', '是', '好看', '的'])
        self.assertListEqual(b, ['m','w', 'n', 'w', 'f', 'nr', 't'])




class test_replace_some_nourns(TestCase):
    def testa(self):
        words = ['咕咕', '《', '游戏', '人生', '》', '是', '好看', '的']
        postags = ['m', 'w', 'n', 'n', 'w', 'f', 'm', 'm']
        a, b, c = replace_some_nourns(words, postags)
        self.assertListEqual(a,['咕咕', '《', 'BOOKNAME', 'BOOKNAME', '》', '是', '好看', '的'])
        self.assertListEqual(b, ['m', 'w', 'n', 'n', 'w', 'f', 'm', 'm'])
        self.assertDictEqual(c, {2: '游戏', 3: '人生'})
        
    def testb(self):
        words = ['咕咕', '?', '游戏', '人生', '?', '是', '好看', '的']
        postags = ['m', 'w', 'nr', 'nr', 'w', 'f', 'm', 'm']
        a, b, c = replace_some_nourns(words, postags)
        self.assertListEqual(a,['咕咕', '?', 'NAME', 'NAME', '?', '是', '好看', '的'])
        self.assertListEqual(b, ['m', 'w', 'nr', 'nr', 'w', 'f', 'm', 'm'])
        self.assertDictEqual(c, {2: '游戏', 3: '人生'})
    
    