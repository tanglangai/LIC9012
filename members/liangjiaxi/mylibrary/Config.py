"""
梁家熙
放一些公共的配置文件类似的东西
注意千万不要循环引用！！！
"""
from common_utils.Pipeline import Pipeline
from common_utils.preprocess_merge_something import merge_adjoint_postag, \
merge_name_punctuation, merge_quotation_marks, replace_some_nourns
from common_utils.post_process_filter import filter_only_one_character

class Config(object):
    # 视图存放的位置
    scheme_pth = "/home/liangjx/teamextraction/data/processed_data/human_schemes"
    
    # 要保留的词汇，为了在attention计算的时候使用
    RESERVED_LSIT = ['nr', 'nw', 'nt', 'nz', 'ns', 't', 'n', 'm']

    pre_pipeline = Pipeline([
        merge_name_punctuation,
        merge_adjoint_postag,
        merge_quotation_marks,
        replace_some_nourns
    ])
    post_pipeline = Pipeline(
        [
            filter_only_one_character
        ]
    )