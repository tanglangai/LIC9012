from NN_models.bilstm import Bilstm
from NN_models.dgcnn import Dgcnn
from NN_models.re_model import Pcnn
from NN_models.joint_model import JModel
from NN_models.bilstm_crf import BilstmCRF
from dataloader.batch_iterator import Create_Iterator
from dataloader.dealing_with_mongo import MongoAgent

import tensorflow as tf
from random import shuffle
import os
import numpy as np
from typing import List
import time
from sklearn.metrics import precision_recall_fscore_support

class Jframe:

    def __init__(self, sess = None, nn_model = None, config = None, summary = True, summary_dir = './log'):
        self.sess = sess
        self.best_f1 = 0
        self.config = config
        self.model_name = nn_model
        if nn_model is None or nn_model == 'pcnn':
            self.nn = Pcnn(config=config)
        elif nn_model == 'joint':
            self.nn = JModel(config=config)
        else:
            raise NotImplementedError
        self.nstep_summary = 0
        self.summary = summary
        self.summary_dir = summary_dir
        self.sess.run(tf.global_variables_initializer())

    def make_one_batch(self, insts, max_length):
        batch_size = len(insts)
        if max_length is None:  # if no max_length is given, set it to be the max length in the batch.
            max_length = 1
            for inst in insts:
                if inst['char_size'] > max_length:
                    max_length = inst['char_size']

        char_array = np.zeros(shape=(batch_size, max_length))
        mask_array = np.zeros(shape=batch_size)
        pos_array = np.zeros(shape=(batch_size, max_length, 2))
        seg_array = np.zeros(shape=(batch_size, max_length, 2))
        ne_array = np.zeros(shape=(batch_size, max_length, 1))
        freq_array = np.zeros(shape=(batch_size, max_length, 1))

        distance_array = np.zeros(shape=(batch_size, max_length, 2))
        is_entity_array = np.zeros(shape=(batch_size, max_length,2), dtype=np.float)
        ner_label_array = np.zeros(shape=(batch_size, max_length, 2), dtype=np.float)
        ner_weight_array = np.zeros(shape=(batch_size, max_length, 2))
        rel_label_array = np.zeros(shape=batch_size)

        for k, inst in enumerate(insts):
            char_size = inst['char_size']
            char_array[k, :char_size] = np.array(inst['char_index'])
            mask_array[k] = char_size
            pos_array[k, :char_size, 0] = np.array(inst['pos_index'])
            pos_array[k, :char_size, 1] = np.array(inst['ltp_pos_index'])
            seg_array[k, :char_size, 0] = np.array(inst['bmes_index'])
            seg_array[k, :char_size, 1] = np.array(inst['ltp_bmes_index'])
            ne_array[k, :char_size, 0] = np.array(inst['ltp_ner_index'])
            freq_array[k, :char_size, 0] = np.array(inst['char_freq'])
            if 'sub_label' in inst:
                ner_label_array[k, :char_size, 0] = np.array(inst['sub_label'])
                ner_label_array[k, :char_size, 1] = np.array(inst['ob_label'])
                ner_weight_array[k, :char_size, 0] = np.array(inst['sub_weight'])
                ner_weight_array[k, :char_size, 1] = np.array(inst['ob_weight'])
            if 'distance2sub' in inst:
                distance_array[k, :char_size, 0] = np.array(inst['distance2sub'])
                distance_array[k, :char_size, 1] = np.array(inst['distance2ob'])
                is_entity_array[k, :char_size, 0] = np.array(inst['is_subject'])
                is_entity_array[k, :char_size, 1] = np.array(inst['is_object'])
            if 'rel_index' in inst:
                rel_label_array[k] = inst['rel_index']

        feed_dict = {self.nn.text: char_array,
                     self.nn.mask: mask_array,
                     self.nn.postag: pos_array,
                     self.nn.bmes: seg_array,
                     self.nn.netag: ne_array,
                     self.nn.freq: freq_array,
                     self.nn.distance: distance_array,
                     self.nn.is_entity: is_entity_array,
                     self.nn.ner_label: ner_label_array,
                     self.nn.ner_weight: ner_weight_array,
                     self.nn.rel_label: rel_label_array}
        return feed_dict


    def create_batch_generator(self, X, batch_size, nepoch, max_length=None, shuffle_bool=True):
        if isinstance(X, MongoAgent):
            list_length = X.collection.count_documents({})
            iter_ = X.collection_iterator(shuffle_bool=shuffle_bool)
            n=0; i=0
            while i < nepoch:
                onebatch = []
                for _ in range(batch_size):
                    if n>= list_length:
                        iter_ = X.collection_iterator(shuffle_bool=shuffle_bool)
                        n=0; i+=1
                    onebatch.append(next(iter_))
                    n+=1
                yield self.make_one_batch(onebatch, max_length=max_length)
        elif isinstance(X, List):
            n=0; i =0; list_length = len(X)
            while i < nepoch:
                onebatch = []
                for _ in range(batch_size):
                    if n >= list_length:
                        if shuffle_bool:
                            shuffle(X)
                        n=0; i+=1
                    onebatch.append(X[n])
                    n+=1
                yield self.make_one_batch(onebatch, max_length=max_length)
        else:
            raise NotImplementedError

    def fit(self, X, mode, dev_X=None, batch_size=None, nepoch=None, save_best=False):
        print('Training...')
        if batch_size is None:
            batch_size = self.config.batch_size
        if nepoch is None:
            nepoch = self.config.nepoch
        if self.summary:
            self.nn.add_stat(mode=mode)
            if os.path.exists(self.summary_dir):
                os.system('rm -rf ' + self.summary_dir)
            self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)

        self.current_epoch = 0
        self.current_batch = 0
        if isinstance(X, List):
            self.n_train = len(X)
        elif isinstance(X, MongoAgent):
            self.n_train = X.collection.count_documents({})
        print('total train sample number: ', self.n_train)
        self.train_iter = self.create_batch_generator(X,
                                                      batch_size = batch_size,
                                                      nepoch = nepoch,
                                                      max_length=None,
                                                      shuffle_bool=True)
        starttime = time.time()
        for fedict in self.train_iter:
            if self.current_batch * batch_size >= self.current_epoch * self.n_train:
                self.current_epoch += 1
                print("\n## The {} Epoch, All {} Epochs ! ##".format(self.current_epoch, nepoch))
                if mode == 'ner':
                    loss = self.sess.run(self.nn.ner_loss, feed_dict = fedict)
                elif mode == 'rel':
                    loss = self.sess.run(self.nn.rel_loss, feed_dict = fedict)
                else:
                    raise NotImplementedError
                print("\nbatch_count = {}, loss is {:.6f}.".format(self.current_batch, loss))
                endtime = time.time()
                print("\nTrain Time {:.3f}".format(endtime - starttime))
                starttime = time.time()

                if (dev_X is not None) and (self.current_batch > 30):
                    print('On train set:')
                    # self.score(X, mode=mode)
                    print('On dev set:')
                    f1 = self.score(dev_X, mode=mode)
                    if save_best:
                        if f1 > self.best_f1:
                            save_name = self.model_name + '-' + str(self.current_epoch)
                            self.save(save_name)
                            self.best_f1 = f1

            self.current_batch += 1

            feed_dict = {self.nn.step: self.current_batch}
            feed_dict.update(fedict)
            if mode == 'rel':
                loss, _ = self.sess.run([self.nn.rel_loss, self.nn.train_step_rel], feed_dict=feed_dict)
            elif mode == 'ner':
                loss, _ = self.sess.run([self.nn.ner_loss, self.nn.train_step_ner], feed_dict=feed_dict)
            else:
                raise NotImplementedError
            if self.current_batch % 5 == 1:
                print("\nbatch_count = {}, loss is {:.6f}.".format(self.current_batch, loss))
            if self.summary:
                stat = self.sess.run(self.nn.stat, feed_dict=feed_dict)
                self.summary_writer.add_summary(stat, self.current_batch)

        return None

    def predict_proba(self, X, mode):
        res = []
        for fedict in self.create_batch_generator(X,
                                                  batch_size=512,
                                                  nepoch=1,
                                                  max_length=self.config.max_sentence_length):
            if mode == 'rel':
                bres = self.sess.run(self.nn.predict_proba_rel, feed_dict=fedict)
            elif mode == 'ner':
                bres = self.sess.run(self.nn.predict_proba_ner, feed_dict=fedict)
            else:
                raise NotImplementedError
            res += list(bres)
        if isinstance(X, List):
            lgth_X = len(X)
        elif isinstance(X, MongoAgent):
            lgth_X = X.collection.count_documents({})
        else:
            raise NotImplementedError
        return res[:lgth_X]

    def predict(self, X, mode):
        res = []
        if mode == 'rel':
            for proba in self.predict_proba(X, mode=mode):
                pred = np.argmax(proba)
                res.append(pred)
        elif mode == 'ner':
            for proba in self.predict_proba(X, mode=mode):
                pred = proba > 0.5
                res.append(pred)
        else:
            raise NotImplementedError
        return res

    def score(self, X, mode):
        preds = self.predict(X, mode)
        if mode == 'rel':
            reals = [inst['relation_index'] for inst in X]
            precision, recall, f1, _ = precision_recall_fscore_support(reals, preds, average='macro')
            print('recall: ', recall)
            print('precision: ', precision)
            print('f1_score: ', f1)
            return f1
        elif mode == 'ner':
            sub_ps = [];
            sub_rs = [];
            sub_f1s = []
            ob_ps = [];
            ob_rs = [];
            ob_f1s = []
            if isinstance(X, MongoAgent):
                samples = X.collection_iterator()
            elif isinstance(X, List):
                samples = X
            else:
                raise NotImplementedError
            for pred, sample in zip(preds, samples):
                size = sample['char_size']
                pred = pred[:size]
                sub_real = sample['sub_label']
                ob_real = sample['ob_label']
                sub_precision, sub_recall, sub_f1, _ = precision_recall_fscore_support(sub_real, pred[:, 0],
                                                                                       average='binary')
                ob_precision, ob_recall, ob_f1, _ = precision_recall_fscore_support(ob_real, pred[:, 1],
                                                                                    average='binary')
                sub_ps.append(sub_precision)
                sub_rs.append(sub_recall)
                sub_f1s.append(sub_f1)
                ob_ps.append(ob_precision)
                ob_rs.append(ob_recall)
                ob_f1s.append(ob_f1)
            sub_mean_f1 = np.mean(sub_f1s)
            ob_mean_f1 = np.mean(ob_f1s)
            mean_f1 = (sub_mean_f1+ob_mean_f1)/2.0
            print('Subject Average recall: ', np.mean(sub_rs))
            print('Subject Average precision: ', np.mean(sub_ps))
            print('Subject Average f1_score: ', sub_mean_f1)
            print('Object Average recall: ', np.mean(ob_rs))
            print('Object Average precision: ', np.mean(ob_ps))
            print('Object Average f1_score: ', ob_mean_f1)
        else:
            raise NotImplementedError
        return mean_f1

    def save(self, name, save_dir = './save'):
        os.makedirs(save_dir, exist_ok = True)
        checkpoint_path = save_dir + '/{}.ckpt'.format(name)
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path, global_step=None)  # save model

    def restore(self, name, var_list=None, save_dir='./save'):
        checkpoint_path = save_dir + '/{}.ckpt'.format(name)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(self.sess, checkpoint_path)
