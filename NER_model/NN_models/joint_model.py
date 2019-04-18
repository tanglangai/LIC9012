from NN_models.basic import BasicDeepLearningModel
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class JModel(BasicDeepLearningModel):

    def build(self):
        self.add_placeholders()
        self.add_body()
        self.add_loss()
        self.add_optimizer()
        #self.add_stat()

    def add_placeholders(self):
        # using dynamic rnn, we don't need to fix sentence size
        with tf.variable_scope('placeholders') as scope:
            # =====common text feature===================
            self.text = tf.placeholder(tf.int32, shape=[None, None], name = 'text_placeholder')
            self.mask = tf.placeholder(tf.int32, shape=[None], name = 'mask_placeholder')
            self.step = tf.placeholder(tf.int32)

            self.postag = tf.placeholder(tf.int32, shape=[None, None, 2], name = 'pos_placeholder')
            self.bmes = tf.placeholder(tf.int32, shape=[None, None, 2], name = 'bmes_placeholder')
            self.netag = tf.placeholder(tf.int32, shape=[None, None, 1], name = 'ne_placeholder')
            self.freq = tf.placeholder(tf.float64, shape=[None, None, 1], name = 'freq_placeholder')

            # ============entity position feature for relation extraction==============
            self.distance = tf.placeholder(dtype=tf.int32, shape=[None, None, 2], name='distance_to_entity')
            self.is_entity = tf.placeholder(dtype=tf.float64, shape=[None, None, 2], name='is_entity')

            # ============ner and rel labels===========================================
            self.ner_label = tf.placeholder(tf.float64, shape=[None, None, self.config.label_vocab_size], name='ner_label_placeholder')
            self.ner_weight = tf.placeholder(tf.float64, shape=[None, None, self.config.label_vocab_size], name='ner_weight_placeholder')
            self.rel_label = tf.placeholder(tf.int32, shape=[None], name='relation_label_placeholder')

    def add_body(self):
        with tf.variable_scope('Body') as _:
            with tf.variable_scope('Embeding') as _:
                with tf.variable_scope('Char_Embeding') as _:
                    embed_init = np.random.random((self.config.char_vocab_size, self.config.embed_dim))
                    self.embed_matrix = tf.Variable(initial_value=embed_init, name = 'embed_matrix')
                    char_embedded = tf.nn.embedding_lookup(self.embed_matrix, self.text) #[None, sentence_length, embed_dim]

                with tf.variable_scope('Postag_Embeding') as _:
                    embed_init = np.random.random((self.config.pos_vocab_size, self.config.pos_embed_dim))
                    self.pos_embed_matrix = tf.Variable(initial_value=embed_init, name='pos_embed_matrix')
                    pos_embedded = tf.nn.embedding_lookup(self.pos_embed_matrix, self.postag)
                    # [None, sentence_length, 2, pos_embed_dim]
                    pos_embedded = tf.concat([pos_embedded[:,:,0,:], pos_embedded[:,:,1,:]], axis = -1)

                with tf.variable_scope('Seg_Embeding') as _:
                    seg_embed_matrix = tf.constant(np.eye(4), dtype = tf.float64)
                    seg_embedded = tf.nn.embedding_lookup(seg_embed_matrix, self.bmes)
                    # [None, sentence_length, 2, 4]
                    # batch_shape = seg_embedded.get_shape().as_list()
                    # seg_embedded = tf.reshape(seg_embedded, [batch_shape[0], batch_shape[1], -1])
                    seg_embedded = tf.concat([seg_embedded[:, :, 0, :], seg_embedded[:, :, 1, :]], axis=-1)

                with tf.variable_scope('NEtag_Embeding') as _:
                    embed_init = np.random.random((self.config.ne_vocab_size, self.config.ne_embed_dim))
                    self.ne_embed_matrix = tf.Variable(initial_value=embed_init, name='ne_embed_matrix')
                    ne_embedded = tf.nn.embedding_lookup(self.ne_embed_matrix, self.netag)
                    # [None, sentence_length, 1, ne_embed_dim]
                    ne_embedded = tf.squeeze(ne_embedded, axis=2)

            with tf.variable_scope('Entity_Embeding'):
                self.position_embed_matrix = tf.get_variable('entity_position_embedding',
                                                             [600, self.config.position_embed_dim], dtype=tf.float64,
                                                             initializer=tf.contrib.layers.xavier_initializer())
                distance_embedded = tf.nn.embedding_lookup(self.embed_matrix, self.distance)
                distance_embedded = tf.concat([distance_embedded[:, :, 0, :], distance_embedded[:, :, 1, :]], axis=-1)
                entity_enbedded = tf.concat([distance_embedded, self.is_entity], axis=-1)

            with tf.variable_scope('Concat') as _:
                #self.freq = tf.nn.tanh(self.freq - 1000)
                #input_embedded = tf.concat([char_embedded, pos_embedded, seg_embedded, ne_embedded, self.freq], axis = -1)
                input_embedded = tf.concat([char_embedded, pos_embedded, seg_embedded, ne_embedded], axis=-1)

            with tf.variable_scope('Core') as _:
                # total_embed_dim = input_embedded.get_shape().as_list()[-1]
                # batch_sentence_length = input_embedded.get_shape().as_list()[1]
                # input_embedded = tf.reshape(input_embedded, [-1, total_embed_dim])
                conv = tf.layers.dense(input_embedded, units=128, activation=tf.nn.relu)
                #conv = tf.reshape(conv, [-1, batch_sentence_length, total_embed_dim])

                conv = dgc_block(conv, filters=128, kernel_size=1, dr=1)
                conv = dgc_block(conv, filters=128, kernel_size=3, dr=1)
                conv = dgc_block(conv, filters=128, kernel_size=3, dr=2)
                conv = dgc_block(conv, filters=128, kernel_size=3, dr=4)
                conv = dgc_block(conv, filters=128, kernel_size=1, dr=8)

                core_out = conv

            with tf.variable_scope('Ner_output') as _:
                self.ner_logits = tf.layers.dense(core_out, units=self.config.label_vocab_size)
                if self.config.label_vocab_size == 1:
                    self.ner_logits = tf.squeeze(self.logits, axis = -1) # if label is 1-dim, squeeze logits

                # max_predict_proba = tf.reduce_max(tf.nn.sigmoid(self.logits), axis=1, keepdims=True)
                # self.predict_proba = tf.nn.sigmoid(self.logits)/max_predict_proba

                self.predict_proba_ner = tf.nn.sigmoid(self.ner_logits)

            with tf.variable_scope('Zip_encoding'):
                text_encoded = zip_encode(tf.concat([core_out, distance_embedded], axis=-1), output_dim=128, hidden_dim=128)
                subject_encoded = zip_encode(tf.expand_dims(entity_enbedded[:, :, -2], axis=-1) * core_out, output_dim=64, hidden_dim=64)
                object_encoded = zip_encode(tf.expand_dims(entity_enbedded[:, :, -1], axis=-1) * core_out, output_dim=64, hidden_dim=64)
                tri_encoded = tf.concat([text_encoded, subject_encoded, object_encoded], axis=-1) # batch_size*256

            with tf.variable_scope('Rel_output') as _:
                x_train, x_test = dense_with_dropout(train=tri_encoded, test=tri_encoded, units=128, activation=tf.nn.relu, dropout_keep=0.75)
                x_train, x_test = dense_with_dropout(train=x_train, test=x_test, units=64, activation=tf.nn.relu, dropout_keep=0.75)
                x_train = tf.concat([tri_encoded, x_train], axis=-1)
                x_test = tf.concat([tri_encoded, x_test], axis=-1)


                self.rel_train_logit, self.rel_test_logit = dense_with_dropout(train=x_train,
                                                                               test=x_test,
                                                                               units=self.config.n_relation,
                                                                               activation=None,
                                                                               dropout_keep=0.75)

                self.predict_proba_rel = tf.nn.softmax(self.rel_test_logit)

    def add_loss(self):
        with tf.variable_scope('ner_Losses') as _:
            self._losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ner_label, logits=self.ner_logits)
            self._losses = self._losses * self.ner_weight
            self.ner_loss = tf.reduce_mean(self._losses)

        with tf.variable_scope('rel_Losses') as _:
            self.rel_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.rel_label, logits=self.rel_train_logit)

    def add_optimizer(self):
        self.learning_rate = 0.0003 + tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                                 global_step=self.step,
                                                                 decay_steps=1000,
                                                                 decay_rate=2.1 / 2.713)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                self.config.adam_beta1,
                                                self.config.adam_beta2)
        self.train_step_ner = self.optimizer.minimize(self.ner_loss)
        self.train_step_rel = self.optimizer.minimize(self.rel_loss)

    def add_stat(self, mode):
        with tf.variable_scope('stats'):
            if mode == 'rel':
                tf.summary.scalar('rel_loss', self.rel_loss)
            elif mode == 'ner':
                tf.summary.scalar('ner_loss', self.ner_loss)
            else:
                raise NotImplementedError
            tf.summary.scalar('lr', self.learning_rate)
            self.stat = tf.summary.merge_all()


def dgc_block(input, filters, kernel_size, dr, pd='same'):
    """
    gated dilation conv1d layer
    """
    glu = tf.sigmoid(tf.layers.conv1d(input,
                                      filters,
                                      kernel_size,
                                      dilation_rate=dr,
                                      padding=pd,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      bias_initializer=tf.contrib.layers.xavier_initializer()))
    conv = tf.layers.conv1d(input, filters, kernel_size, dilation_rate=dr, padding=pd,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.contrib.layers.xavier_initializer())
    gated_conv = tf.multiply(conv, glu)
    gated_x = tf.multiply(input, 1 - glu)
    outputs = tf.add(gated_x, gated_conv)
    return outputs

def zip_encode(input, output_dim, hidden_dim = 16):
    """
    zip encode the input
    :param input: [batch_size, sentence_length, k]
    :return: [batch_size, output_dim]
    """
    attention = tf.layers.conv1d(input, filters=hidden_dim, kernel_size = 1, activation=tf.nn.relu)
    attention = tf.layers.conv1d(attention, filters=1, kernel_size=1, activation=tf.tanh) #[batch_size, sentence_length, 1]
    res = tf.layers.conv1d(input, filters=output_dim, kernel_size=1, activation=None)
    res = tf.reduce_mean(res * attention, axis=1)
    return res

def dense_with_dropout(train, test, units, dropout_keep, activation=None):
    init_W = np.random.random((units, train.shape[1]))
    W_ = tf.Variable(init_W, dtype=tf.float64)
    init_b = np.random.random(units)
    bias = tf.Variable(init_b, dtype=tf.float64)
    train_logit = tf.matmul(train, tf.transpose(W_)) + bias
    test_logit = tf.matmul(test, tf.transpose(W_)) + bias
    res_train = tf.contrib.layers.dropout(train_logit, keep_prob=dropout_keep)
    res_test = test_logit
    return res_train, res_test

