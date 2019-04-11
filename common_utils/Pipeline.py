"""
梁家熙
想定义一个类似管道的，
数据清理的流程
"""
from typing import Callable, Dict, List


class Pipeline(object):
    """
    定义提取的函数。
    """
    def __init__(self, function_list: List = None):
        if function_list:
            assert isinstance(function_list, list)
            self.function_lst = function_list
        else:
            self.function_lst = []
            
    def append_func(self, func: Callable) -> None:
        if not callable(func):
            raise TypeError
        self.function_lst.append(func)
    
    def run(self, *args) -> Dict:
        
        #我们约定数据能够输入输出是一个格式的，这样才能不断调用函数。
        for func in self.function_lst:
            args = func(*args)

        return args
    
    def __call__(self, *arg):
        return self.run(*arg)

if __name__ == '__main__':
    a=lambda x: x[0]
    p = Pipeline()
    p.append_func(a)
    