import os
import json
import itertools
import pickle
from tqdm import tqdm

from scripts.misc import Char_Feature_Pipeline
from scripts.misc import json_iter, get_vocabulary_and_maxlgth




def output_char_feature_data(data_dir, pyltp_path, output_dir = './processed_data'):
    """
    run the whole pipeline to generate char feature data and save them in output_dir
    :param data_dir: input data direction, must contain files: train_data.json, dev_data.json, test1_data_postag.json
    :param output_dir: output data direction, default to ./processed_data
    :return: None
    """

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train_data.json')
    dev_path = os.path.join(data_dir, 'dev_data.json')
    test_path = os.path.join(data_dir, 'test1_data_postag.json')

    def join_data_loader(paths):
        raw_iter = itertools.chain.from_iterable([json_iter(path) for path in paths])
        return raw_iter
    raw_iter = join_data_loader([train_path, dev_path, test_path])

    # get char vocabulary, char frequency and max sentence length
    char_voca, counter, max_length = get_vocabulary_and_maxlgth(tqdm(raw_iter))
    allinone = {'voca': char_voca,
                'word_freq': counter,
                'max_length': max_length}
    pickle_path = os.path.join(output_dir, 'total_voca.pickle')
    with open(pickle_path, 'wb') as f:
        pickle.dump(allinone, f)

    pipeline = Char_Feature_Pipeline(char_voca=char_voca, freq_dict=counter, pyltp_path=pyltp_path)

    train_output = pipeline.pipeline(json_iter(train_path), label_bool=True)
    inst = next(train_output)
    dev_output = pipeline.pipeline(json_iter(dev_path), label_bool=True)
    test_output = pipeline.pipeline(json_iter(test_path), label_bool=False)

    for filename, data in zip(['train.json','dev.json','test.json'], [train_output, dev_output, test_output]):
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            for dct in data:
                json.dump(dct, f)
                f.write('\n')
    pipeline.release_pyltp_model()

    return None


if __name__ == '__main__':

    DATA_DIR = '/storage/gs2018/Competition/LIC-IE/data/origin_data'
    CESS_DIR = './processed_data'
    PYLTP_PATH = "/storage/Competitions/fddc/renyan/pyltp-model/ltp_data_v3.4.0"

    output_char_feature_data(data_dir = DATA_DIR, output_dir = CESS_DIR, pyltp_path=PYLTP_PATH)

    print('exit')




