from allennlp.common.testing import ModelTestCase
from allennlp.common.util import import_submodules

class DecoderTest(ModelTestCase):
    def setUp(self):
        super(DecoderTest, self).setUp()
        import_submodules('members.liangjiaxi.mylibrary')
        self.set_up_model(
            '/home/liangjx/teamextraction/members/liangjiaxi/experiment/decoder_baseline.json',
            '/home/liangjx/teamextraction/data/processed_data/fk_trn.json'
        )
        
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load((self.param_file))