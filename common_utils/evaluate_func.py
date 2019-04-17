"""
用于评价
除了英文字母小写外，其他严格等于
"""
from typing import List, Dict
import demjson


class EvaluatePrediction(object):
    """
    用法如下
    eva = EvaluatePrediction()
    eva.evluate('./predict.json','./golden.json')
    """
    def __init__(self):
        self.right_num_spo = 0
        self.all_predict_num_spo = 0
        self.should_have_num_spo = 0
        
    def evluate(self, predict_file_pth: str, golden_file_pth: str)->None:
        """
        :param predict_file: 预测的文件路径
        :param golden_file: 真实的文件路径
        :return:
        """
        golden_file = open(golden_file_pth)
        predict_file = open(predict_file_pth)
        i = 0
        while True:
            i += 1
            
            if i%100 ==0:
                print("i is {}".format(i))
                print(self.right_num_spo)
                print(self.all_predict_num_spo)
                print(self.should_have_num_spo)
                
            golden_line = golden_file.readline()
            predict_line = predict_file.readline()
            
            if (len(golden_line) == 0) or (len(predict_line) == 0):
                break

            golden_spo_text = demjson.decode(golden_line)['text']
            predict_spo_text = demjson.decode(predict_line)['text']
            j = 0
            
            # 如果预测的时候跳过了噪音行，那么这里就要进行对其处理
            while golden_spo_text != predict_spo_text:
                golden_line = golden_file.readline()
                golden_spo_text = demjson.decode(golden_line)['text']
                j += 1
                if j>200:
                    raise Exception('没有找到答案')
                if (len(golden_line) == 0) or (len(predict_line) == 0):
                    break
            
            golden_spo_list = demjson.decode(golden_line)['spo_list']
            predict_spo_list = demjson.decode(predict_line)['spo_list']

            right_num_spo, all_predict_num_spo, should_have_num_spo = self.get_pre_recall_f1(golden_spo_list,
                                                                                         predict_spo_list)
            self.right_num_spo += right_num_spo
            self.all_predict_num_spo += all_predict_num_spo
            self.should_have_num_spo += should_have_num_spo
            
        if len(golden_line) != 0 or len(predict_line) != 0:
            print("预测文件行数与真实文件不一致")
        
        
        golden_file.close()
        predict_file.close()
        
        precision = self.right_num_spo / self.all_predict_num_spo
        recall = self.right_num_spo / self.should_have_num_spo
        f1 = 2*precision*recall / (precision + recall)
        print("precision is {}, recall is {}, f1 is {}".format(precision, recall, f1))
        
    def get_pre_recall_f1(self,golden_spo_list: List[Dict[str, str]],
                          predict_spo_list: List[Dict[str, str]]) -> [int, int, int]:
        """
            对比两个列表，判断预测正确数目，所有检测的数目，应包含数目
        """
        assert len(golden_spo_list) != 0 and len(predict_spo_list) != 0
        all_predict_num_spo = len(predict_spo_list)
        should_have_num_spo = len(golden_spo_list)
        right_num_spo = 0
        for golden_spo in golden_spo_list:
            for predict_spo in predict_spo_list:
                golden_predicate = golden_spo['predicate'].lower()
                golden_subject = golden_spo['subject'].lower()
                golden_object = golden_spo['object'].lower()
                predict_predicate = predict_spo['predicate'].lower()
                predict_subject = predict_spo['subject'].lower()
                predict_object = predict_spo['object'].lower()
                
                if golden_predicate == predict_predicate and \
                        golden_subject == predict_subject and \
                        golden_object == predict_object:
                    right_num_spo += 1
                    
        return right_num_spo, all_predict_num_spo, should_have_num_spo
        
    
if __name__ == '__main__':
    a = EvaluatePrediction()
    a.evluate('./dev_data_4_17.json','../data/origin_data/dev_data.json')