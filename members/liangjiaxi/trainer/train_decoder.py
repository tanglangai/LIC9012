from allennlp.common import Params
from allennlp.common.util import import_submodules
# from members.liangjiaxi.mylibrary.modified.modified_import import import_submodules
from members.liangjiaxi.mylibrary.modified.modified_train import train_model
import shutil
import visdom

from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
import warnings
warnings.filterwarnings('ignore')
import torch
import random
import numpy as np
seed = 27
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

#相对导入的技巧  https://blog.csdn.net/xie_0723/article/details/78004649
# decoder baseline
# import_submodules('members.liangjiaxi.mylibrary')
# params = Params.from_file('../experiment/decoder_baseline.json')
# serialization_dir = '../../../tmp/model_weight/decoder_4_15'

#  替换matrix，变成全连接
import_submodules('members.liangjiaxi.mylibrary')
params = Params.from_file('../experiment/decoder_multi.json')
serialization_dir = '../../../tmp/model_weight/decoder_multi_4_15'

model = train_model(
                    params,
                    serialization_dir,
                    force=True,
                    )