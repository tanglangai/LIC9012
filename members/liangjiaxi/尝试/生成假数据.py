

"""数据格式
words: List[str]
poses: List[str]  这两部分假设代表的是我们从上层获得的信息。

entity_pair: List[int, int] 代表的是位置
label: int，代表这对词属于51种视图中的哪一个
"""

from common_utils.scheme_mapping import scheme2index
from common_utils.scheme_mapping import generaterows
scheme_pth = r'/home/liangjx/teamextraction/data/original_data/human_schemes'

sch2ind = scheme2index(scheme_pth)
import demjson
demjson.decode()
