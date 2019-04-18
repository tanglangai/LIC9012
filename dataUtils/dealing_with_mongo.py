from pymongo import MongoClient
from tqdm import tqdm
from random import shuffle

class MongoAgent():

    def __init__(self, port = 'localhost:27017', db_name = 'PIL9102', collection_name = None):
        self.conn = MongoClient(port)
        self.db = self.conn[db_name]
        if collection_name is None:
            self.collection = None
        else:
            assert collection_name in self.db.list_collection_names()
            self.collection = self.db[collection_name]

    def upload2mongo(self, data):
        collection = self.collection
        for i, dct in enumerate(tqdm(data)):
            if '_id' not in dct:
                dct.update({'_id': i})
            try:
                collection.insert_one(dct)
            except:
                collection.save(dct)

    def update_mongo(self, data):
        for dct in data:
            self.single_update_mongo(dct=dct)

    def single_update_mongo(self, dct):
        assert '_id' in dct
        self.collection.update_one(filter={'_id': dct['_id']}, update={'$set': dct})

    def single_upload2mongo(self, dct):
        assert '_id' in dct
        try:
            self.collection.insert_one(dct)
        except:
            self.collection.save(dct)

    def load_by_id(self, Id):
        res = self.collection.find_one({'_id': Id})
        return res

    def list_collections(self):
        names = self.db.list_collection_names()
        return names

    def set_collection(self, collection_name):
        assert collection_name in self.list_collections()
        self.collection = self.db[collection_name]

    def create_collection(self, collection_name):
        assert collection_name not in self.list_collections()
        self.collection = self.db[collection_name]

    def collection_iterator(self, collection_name = None, nsample = None, shuffle_bool=False):
        if collection_name is None:
            pass
        else:
            self.set_collection(collection_name)
        if nsample is None:
            total_num = self.collection.count_documents({})
        else:
            total_num = nsample
        IDs = list(range(total_num))
        if shuffle_bool:
            shuffle(IDs)
        for ID in IDs:
            sample = self.load_by_id(Id=ID)
            yield sample




if __name__ == '__main__':

    mongoAgent = MongoAgent()
    mongoAgent.list_collections()

    sample = mongoAgent.load_by_id(5, 'train_data')

    print('exit')