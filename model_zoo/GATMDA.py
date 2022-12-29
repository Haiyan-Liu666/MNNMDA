import time
import numpy as np
import random
import scipy.sparse as sp
import sys
import os
sys.path.append("./model_zoo/GATMDA-master/src")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
try:
    import tensorflow as tf
    from models import GAT
    from inits import adj_to_bias, sparse_matrix, normalize_features
    # from inits import test_negative_sample
    from metrics import masked_accuracy
    # from metrics import ROC
except:
    print("GATMDA environment error!")

from model_zoo.base import BaseModel


def test_negative_sample(indices, N, shape):
    num = 0
    nd,nm = shape
    A = sp.csr_matrix((indices[:,2],(indices[:,0], indices[:,1])),shape=(nd,nm)).toarray()
    mask = np.zeros(A.shape)
    test_neg=np.zeros((N,2))
    while(num<N):
        a = random.randint(0,nd-1)
        b = random.randint(0,nm-1)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg.astype(int)


def ROC(matrix, test_indices, neg_sample_index):
    test_indices = test_indices[test_indices[:, 2]>0.5]
    pos_score = matrix[(test_indices[:, 0], test_indices[:, 1])]
    neg_score = matrix[(neg_sample_index[:, 0], neg_sample_index[:, 1])]
    scores = np.concatenate([pos_score, neg_score])
    labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
    return labels, scores


def generate_mask(labels, N, shape):
    num = 0
    # nd = np.max(labels[:, 0])
    # nm = np.max(labels[:, 1])
    # nd = nd.astype(np.int32)
    # nm = nm.astype(np.int32)
    nd, nm = shape
    A = sp.csr_matrix((labels[:, 2], (labels[:, 0], labels[:, 1])), shape=(nd, nm)).toarray()
    mask = np.zeros(A.shape)
    label_neg = np.zeros((1 * N, 2))
    while (num < 1 * N):
        a = random.randint(0, nd - 1)
        b = random.randint(0, nm - 1)
        if A[a, b] != 1 and mask[a, b] != 1:
            mask[a, b] = 1
            label_neg[num, 0] = a
            label_neg[num, 1] = b
            num += 1
    mask = np.reshape(mask, [-1, 1])
    return mask, label_neg


def load_data(train_indices, test_indices, d_feature, m_feature, shape):
    nd, nm = shape
    logits_train = sp.csr_matrix((train_indices[:, 2], (train_indices[:, 0], train_indices[:, 1])), shape=(nd, nm)).toarray()
    logits_test = sp.csr_matrix((test_indices[:, 2], (test_indices[:, 0], test_indices[:, 1])), shape=(nd, nm)).toarray()

    M = np.copy(np.array(logits_train)) #原本使用完整的interaction
    labels = np.copy(train_indices[train_indices[:,-1]==1])  #原本使用完整的interaction

    # M = logits_train+logits_test
    # labels = np.concatenate([train_indices, test_indices])

    logits_train = logits_train.reshape([-1, 1])
    logits_test = logits_test.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])

    interaction = np.vstack((np.hstack((np.zeros(shape=(nd, nd), dtype=int), M)),
                             np.hstack((M.transpose(), np.zeros(shape=(nm, nm), dtype=int)))))

    F1 = d_feature
    F2 = m_feature
    features = np.vstack((np.hstack((F1, np.zeros(shape=(F1.shape[0], F2.shape[1]), dtype=int))),
                          np.hstack((np.zeros(shape=(F2.shape[0], F1.shape[0]), dtype=int), F2))))
    features = normalize_features(features)
    return interaction, features, sparse_matrix(logits_train), logits_test, train_mask, test_mask, labels


class GATMDA(BaseModel):
    def __init__(self, batch_size, nb_epochs, lr, l2_coef, weight_decay, hid_units, n_heads, residual, **kwargs):
        super(GATMDA, self).__init__(**kwargs)
        self.model_cls = GAT
        self.nonlinearity = tf.nn.elu
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.l2_coef = l2_coef
        self.weight_decay = weight_decay
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.residual = residual
        print('----- Opt. hyperparams -----')
        print('lr: ' + str(lr))
        print('l2_coef: ' + str(l2_coef))
        print('----- Archi. hyperparams -----')
        print('nb. layers: ' + str(len(hid_units)))
        print('nb. units per layer: ' + str(hid_units))
        print('nb. attention heads: ' + str(n_heads))
        print('residual: ' + str(residual))
        print('nonlinearity: ' + str(self.nonlinearity))
        print('model: ' + str(self.model_cls))

    @staticmethod
    def add_argparse_args(parent_parser):
        parent_parser.add_argument("--batch_size", default=1, type=int)
        parent_parser.add_argument("--nb_epochs", default=200, type=int)
        parent_parser.add_argument("--lr", default=0.005, type=float)
        parent_parser.add_argument("--l2_coef", default=0.0005, type=float)
        parent_parser.add_argument("--weight_decay", default=5e-4, type=float)
        parent_parser.add_argument("--residual", default=False, action="store_true")
        parent_parser.add_argument("--hid_units", default=[8], type=int, nargs="+")
        parent_parser.add_argument("--n_heads", default=[4, 1], type=int, nargs="+")
        return parent_parser

    def train_eval(self, train_indices, test_indices, d_feature, m_feature, shape):
        batch_size = self.batch_size
        model = self.model_cls
        hid_units = self.hid_units
        n_heads = self.n_heads
        residual = self.residual
        nonlinearity = self.nonlinearity
        weight_decay = self.weight_decay
        lr = self.lr
        l2_coef = self.l2_coef
        nb_epochs = self.nb_epochs

        interaction, features, y_train, y_test, train_mask, test_mask, labels = load_data(train_indices, test_indices, d_feature, m_feature, shape)
        nb_nodes = features.shape[0]
        ft_size = features.shape[1]

        features = features[np.newaxis]
        interaction = interaction[np.newaxis]
        biases = adj_to_bias(interaction, [nb_nodes], nhood=1)


        nd, nm = shape
        # nd = np.max(labels[:, 0])
        # nm = np.max(labels[:, 1])
        # nd = nd.astype(np.int32)
        # nm = nm.astype(np.int32)
        entry_size = nd * nm
        with tf.Graph().as_default():
            with tf.name_scope('input'):
                feature_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
                bias_in = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
                lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(entry_size, batch_size))
                attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
                ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
                is_train = tf.compat.v1.placeholder(dtype=tf.bool, shape=())

            final_embedding, coefs = model.encoder(feature_in, nb_nodes, is_train,
                                                   attn_drop, ffd_drop,
                                                   bias_mat=bias_in,
                                                   hid_units=hid_units, n_heads=n_heads,
                                                   residual=residual, activation=nonlinearity)
            scores = model.decoder(final_embedding, nd)

            loss = model.loss_sum(scores, lbl_in, msk_in, neg_msk, weight_decay, coefs, final_embedding)

            accuracy = masked_accuracy(scores, lbl_in, msk_in, neg_msk)

            train_op = model.training(loss, lr, l2_coef)

            init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())

            with tf.compat.v1.Session() as sess:
                sess.run(init_op)

                train_loss_avg = 0
                train_acc_avg = 0

                for epoch in range(nb_epochs):

                    t = time.time()

                    ##########    train     ##############

                    tr_step = 0
                    tr_size = features.shape[0]

                    # neg_mask, label_neg = generate_mask(labels, len(train_arr))
                    neg_mask, label_neg = generate_mask(labels, int(train_mask.sum()), shape)

                    while tr_step * batch_size < tr_size:
                        _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                                                            feed_dict={
                                                                feature_in: features[tr_step * batch_size:(
                                                                                                                      tr_step + 1) * batch_size],
                                                                bias_in: biases[
                                                                         tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                lbl_in: y_train,
                                                                msk_in: train_mask,
                                                                neg_msk: neg_mask,
                                                                is_train: True,
                                                                attn_drop: 0.1, ffd_drop: 0.1})
                        train_loss_avg += loss_value_tr
                        train_acc_avg += acc_tr
                        tr_step += 1
                    print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % (
                    (epoch + 1), loss_value_tr, acc_tr, time.time() - t))

                print("Finish traing.")

                ###########     test      ############

                ts_size = features.shape[0]
                ts_step = 0
                ts_loss = 0.0
                ts_acc = 0.0

                print("Start to test")
                while ts_step * batch_size < ts_size:
                    out_come, emb, coef, loss_value_ts, acc_ts = sess.run([scores, final_embedding, coefs, loss, accuracy],
                                                                          feed_dict={
                                                                              feature_in: features[ts_step * batch_size:(
                                                                                                                                    ts_step + 1) * batch_size],
                                                                              bias_in: biases[ts_step * batch_size:(
                                                                                                                               ts_step + 1) * batch_size],
                                                                              lbl_in: y_test,
                                                                              msk_in: test_mask,
                                                                              neg_msk: neg_mask,
                                                                              is_train: False,
                                                                              attn_drop: 0.0, ffd_drop: 0.0})
                    ts_loss += loss_value_ts
                    ts_acc += acc_ts
                    ts_step += 1
                print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

                out_come = out_come.reshape((nd, nm))
                sess.close()
                return out_come

                # return test_labels, score

    def fit_transform(self, train_indices, test_indices, d_feature, m_feature, shape):
        res = self.train_eval(train_indices, test_indices, d_feature, m_feature, shape)
        res = np.array(res)
        return res