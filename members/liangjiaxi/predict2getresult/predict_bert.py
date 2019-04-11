from allennlp.common.util import import_submodules
from allennlp.models import load_archive
from allennlp.predictors import Predictor
import json

from common_utils.scheme_mapping import postag2wordpos
from common_utils.predict_func import tojsonfile, convert_ans_to_original_word
from copy import deepcopy
from members.liangjiaxi.mylibrary.Config import Config


if __name__ == '__main__':
    
    import_submodules('members.liangjiaxi.mylibrary')
    
    PREDICTOR_NAME = 'Bert51baseline_predictor'
    # MODEL_PTH = '/home/liangjiaxi/Projects/extract_information/tmp/finetune'
    MODEL_PTH = '../../../tmp/model_weight/bert51baseline_4_111'
    
    archive = load_archive(MODEL_PTH)
    predictor = Predictor.from_archive(archive, PREDICTOR_NAME)
    
    # TEST_PTH='/home/liangjiaxi/Projects/extract_information/data/processed_data/faketrain.json'
    # TEST_PTH='/home/liangjiaxi/Projects/extract_information/data/origin_data/test1_data_postag.json'
    TEST_PTH = '../../../data/processed_data/faketrain.json'
    
    ans_lst=[]
    i=1
    with open(TEST_PTH, 'r') as file:
        while True:
            #最原始，最简陋的进度条
            print(i)
            i += 1
            line = file.readline()
            if not line:
                break
            

            data = json.loads(line)
            postag = data['postag']
            if not postag:
                continue
            words, poses = postag2wordpos(postag)
            
            p = Config.pre_pipeline
            words, poses = p.run(words, poses)
           
            from common_utils.preprocess_merge_something import replace_some_nourns

            words, poses, position_dict = replace_some_nourns(words, poses)
            
            ans = predictor.predict_json(data)
            ans = convert_ans_to_original_word(ans, words, position_dict)
            ans_lst.append(deepcopy(ans))
  
    # print(ans_lst)
    print("预测完所有的测试集")
    
    #这种写法是错误的，因为要返回的是json文件格式，而这种写入方式会报错！！
    tojsonfile( '../../../tmp/model_weight/faketrainresult.json',ans_lst)
    
