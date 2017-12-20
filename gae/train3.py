from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE, RecommenderOptimizerAE
from input_data import load_data, load_data_from_citeulike, load_data_from_sub_citeulike, load_data_for_mini_batch
from model import GCNModelAE, GCNModelVAE, RecommenderGCNModelAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 50, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'recommender_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('a', 1.0, 'rated')
flags.DEFINE_float('b', 0.01, 'not rated')
flags.DEFINE_integer('num_u', 1500, 'number of users')  # 5551
flags.DEFINE_integer('num_v', 16980, 'number of items')  # 16980

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
# adj, features = load_data(dataset_str)
# adj_citeulike = load_data_from_sub_citeulike()

# Store original adjacency matrix (without diagonal entries) for later
# adj_orig = adj_citeulike
# adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
# adj_orig.eliminate_zeros()


# adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
# adj = adj_train

# print(adj)
# if FLAGS.features == 0:
#    features = sp.identity(adj_citeulike.shape[0])  # featureless features = sp.identity(adj.shape[0])

# Some preprocessing
# adj_norm = preprocess_graph(adj_citeulike)        #some problem, some rows are zeros totally

interaction = load_data_for_mini_batch()

# Define placeholders
placeholders = {
    'features_user': tf.sparse_placeholder(tf.float32),
    'features_item': tf.sparse_placeholder(tf.float32),
    'adj_norm_user': tf.sparse_placeholder(tf.float32),
    'adj_norm_item': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'rate_a_b': tf.placeholder(tf.float32),
    'features_nonzero_user': tf.placeholder(tf.int64),
    'features_nonzero_item': tf.placeholder(tf.int64)
}

#num_nodes = adj_citeulike.shape[0]

#features = sparse_to_tuple(features.tocoo())

#print(num_features)

# Create model
model = None
if model_str == 'gcn_ae':
    pass
    # model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    pass
    # model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'recommender_ae':
    model = RecommenderGCNModelAE(placeholders, input_dim_users=5551, input_dim_items=16980)


# pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'recommender_ae':
        opt = RecommenderOptimizerAE(preds=model.reconstructions,
                                     labels=placeholders['adj_orig'],
                                     a=FLAGS.a, b=FLAGS.b, num_u=FLAGS.num_u, num_v=FLAGS.num_v,
                                     C=placeholders['rate_a_b'])
    '''
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
    '''

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

'''
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
'''

def read_user(f_in='cf-train-1-users.dat', num_u=5551, num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u, num_v)))
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i, int(seg)] = 1
    return R


def cal_precision(p, cut, R):
    R_true = read_user('cf-test-1-users-subb.dat', num_u=2500, num_v=16980)
    dir_save = 'cdl' + str(p)
    # U = np.mat(np.loadtxt(dir_save+'/final-U.dat'))
    # V = np.mat(np.loadtxt(dir_save+'/final-V.dat'))
    num_u = R.shape[0]
    print(num_u)
    num_hit = 0
    # fp = open(dir_save+'/rec-list.dat','w')
    for i in range(num_u):
        if i != 0 and i % 100 == 0:
            print('Iter ' + str(i))

        l_score = np.ravel(R[i, :]).tolist()
        ll_score = R[i, :].tolist()
        # print '*************************'
        # print l_score []
        # print ll_score [[]]

        pl = sorted(enumerate(l_score), key=lambda d: d[1], reverse=True)

        # print pl


        l_rec = list(zip(*pl)[0])[:cut]
        # print l_rec

        s_rec = set(l_rec)
        s_true = set(np.ravel(np.where(R_true[i, :] > 0)[1]))
        # print s_true

        if len(s_true) > 0:
            cnt_hit = len(s_rec.intersection(s_true))

            num_hit += float(cnt_hit) / float(len(s_true))
            # print num_hit
            # fp.write('%d:' % i)
            # fp.write('%d:' % cnt_hit)
            # fp.write(' '.join(map(str,l_rec)))
            # fp.write('\n')
    # fp.close()
    print('Precision: %.3f' % (float(num_hit) / float(num_u)))


cost_val = []
acc_val = []
val_roc_score = []

# adj_label = adj_citeulike
# adj_label = sparse_to_tuple(adj_label)
# tt = adj_orig.tocoo().todense()
# tt = np.array(tt)

# Train model
for epoch in range(FLAGS.epochs):
    arr1 = np.arange(5551)
    arr2 = np.arange(16980)
    np.random.shuffle(arr1)
    np.random.shuffle(arr2)
    indexI_batch = arr1[:FLAGS.num_u]
    indexJ_batch = arr2[:FLAGS.num_v]

    user_sub = interaction[indexI_batch,:]
    item_sub = interaction.transpose()[indexJ_batch,:]

    if FLAGS.features == 0:
        features_user = sp.identity(user_sub.shape[1])
        features_item = sp.identity(item_sub.shape[1])

    features_user = sparse_to_tuple(features_user.tocoo())
    num_features_user = features_user[2][1]
    features_nonzero_user = features_user[1].shape[0]

    features_item = sparse_to_tuple(features_item.tocoo())
    num_features_item = features_item[2][1]
    features_nonzero_item = features_item[1].shape[0]

    t = time.time()

    # Some preprocessing
    adj_norm_user = preprocess_graph(user_sub)  # some problem, some rows are zeros totally
    adj_norm_item = preprocess_graph(item_sub)

    feed_dict1 = dict()
    feed_dict1.update({placeholders['features_user']: features_user})
    feed_dict1.update({placeholders['adj_norm_user']: adj_norm_user})
    #feed_dict1.update({placeholders['num_features_user']: num_features_user})
    feed_dict1.update({placeholders['features_nonzero_user']: features_nonzero_user})
    feed_dict1.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict1.update({placeholders['features_item']: features_item})
    feed_dict1.update({placeholders['adj_norm_item']: adj_norm_item})
    #feed_dict1.update({placeholders['num_features_item']: num_features_item})
    feed_dict1.update({placeholders['features_nonzero_item']: features_nonzero_item})

    tt = user_sub[:,indexJ_batch].tocoo().todense()
    tt = np.array(tt)

    feed_dict1.update({placeholders['adj_orig']: tt})

    C = np.empty((FLAGS.num_u, FLAGS.num_v))
    C.fill(FLAGS.b)
    print(C.shape)
    C[tt > 0] = FLAGS.a
    feed_dict1.update({placeholders['rate_a_b']: C})

    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, model.reconstructions], feed_dict=feed_dict1)

    # te = sess.run([model.embeddings, model.reconstructions], feed_dict = feed_dict)
    # ss = np.array(te[1])
    # print (ss.shape)
    # print(tf.shape(te[1]))

    # Compute average loss
    avg_cost = outs[1]
    R = outs[2]
    # avg_accuracy = outs[2]

    # roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    # val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "time=", "{:.5f}".format(time.time() - t))
    if (epoch + 1) % 100 == 0:
        cal_precision(4, 300, R)

print("Optimization Finished!")

# roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
# print('Test ROC score: ' + str(roc_score))
# print('Test AP score: ' + str(ap_score))
