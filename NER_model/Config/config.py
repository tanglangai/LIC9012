
_config_dict = {
    'batch_size': 32,
    'epoch': 10,
    'max_sentence_length_allow': 300,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'embed_dim': 200,
    'pos_embed_dim': 5,
    'pos_vocab_size': 39,
    'ne_embed_dim': 5,
    'ne_vocab_size': 13,
    'lstm_cell_dim': 32,
    'char_vocab_size': 6481,
    'learning_rate': 0.005,
    'label_vocab_size': 2,
    'max_sentence_length': 300,
    'n_relation':51,
    'position_embed_dim':5

}



class _Config_container(object):

    def __init__(self, config_dict):
        for k in config_dict:
            self.__setattr__(k, config_dict[k])

config = _Config_container(_config_dict)