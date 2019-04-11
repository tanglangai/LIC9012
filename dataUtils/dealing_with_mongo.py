from pymongo import MongoClient
from tqdm import tqdm

class MongoAgent():

    def __init__(self, port = 'localhost:27017', db_name = 'PIL9102', collection_name = None):
        self.conn = MongoClient(port)
        self.db = self.conn[db_name]
        if collection_name is None:
            self.collection = None
        else:
            self.collection = self.db[collection_name]

    def upload2mongo(self, data, collection_name):
        collection = self.db[collection_name]

        for i, dct in enumerate(tqdm(data)):
            if '_id' not in dct:
                dct.update({'_id': i})
            try:
                collection.insert_one(dct)
            except:
                collection.save(dct)

    def single_upload2mongo(self, dct, collection_name = None):
        if collection_name is None:
            pass
        else:
            self.collection = self.db[collection_name]
        assert '_id' in dct
        try:
            self.collection.insert_one(dct)
        except:
            self.collection.save(dct)

    def load_by_id(self, Id, collection_name):
        collection = self.db[collection_name]
        res = collection.find_one({'_id': Id})
        return res

