class Instance(object):
    def __init__(self):
        """
        SPOs is a list of dicts, each has key: 'predicate', 'object_type', 'subject_type', 'object', 'subject'
        label_index is a list of tuples of length 3, (subject_location, object_location, predicate_index)
        """
        self.raw_text = ''
        self.words = []
        self.POSs = []
        self.word_size = 0
        self.words_index = []
        self.POSs_index = []
        self.SPOs = []
        self.label_index = []
