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
from input_data import load_data, load_data_from_citeulike
from model import GCNModelAE, GCNModelVAE, RecommenderGCNModelAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'recommender_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('a',1.0, 'rated')
flags.DEFINE_float('b',0.01, 'not rated')
flags.DEFINE_integer('num_u', 1024, 'number of users')              #     5551
flags.DEFINE_integer('num_v', 1024, 'number of items')             #     16980

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
adj_citeulike = load_data_from_citeulike()
print ("Finish load data: %f", time.time())
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj_citeulike
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#adj = adj_train

#print(adj)
if FLAGS.features == 0:
    features = sp.identity(adj_citeulike.shape[0])  # featureless features = sp.identity(adj.shape[0])

# Some preprocessing
#adj_norm = preprocess_graph(adj_citeulike)        #some problem, some rows are zeros totally
print("Finish normalizing: %f", time.time())

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'rate_a_b': tf.placeholder(tf.float32),
    'num_features': tf.placeholder(tf.int64),
    'features_nonzero': tf.placeholder(tf.int64)
}

num_nodes = adj_citeulike.shape[0]

#features = sparse_to_tuple(features.tocoo())
#num_features = features[2][1]
#features_nonzero = features[1].shape[0]
print("Tocoo: %f", time.time())
# Create model
model = None
if model_str == 'gcn_ae':
    pass
    #model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    pass
    #model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'recommender_ae':
    model = RecommenderGCNModelAE(placeholders, FLAGS.num_u, FLAGS.num_v)

#pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

print(placeholders['adj_orig'])
print(model.reconstructions)
# Optimizer
print (FLAGS.num_u)
with tf.name_scope('optimizer'):
    if model_str == 'recommender_ae':
        print("hhhhhhhhhhhhhhhh")
        opt = RecommenderOptimizerAE(preds=model.reconstructions,
                          labels=placeholders['adj_orig'],
                          a=FLAGS.a, b=FLAGS.b, num_u=FLAGS.num_u, num_v=FLAGS.num_v, C=placeholders['rate_a_b'])
    #elif model_str == 'gcn_vae':
    #    opt = OptimizerVAE(preds=model.reconstructions,
    #                       labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
    #                                                                   validate_indices=False), [-1]),
    #                       model=model, num_nodes=num_nodes,
    #                       pos_weight=pos_weight,
    #                       norm=norm)

print ("hahahhaha")
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

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


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_citeulike
adj_label = sparse_to_tuple(adj_label)
tt = adj_orig.tocoo().todense()
tt = np.array(tt)


# Train model
iter = 0
for epoch in range(FLAGS.epochs):

    for i in range(5551):
        for j in range(16980):
            if i+FLAGS.num_u < 5551 & j+FLAGS.num_v < 16980:
                features = sp.vstack(sp.csr_matrix(features)[i:i+FLAGS.num_u,:],
                                     sp.csr_matrix(features)[5551+j:j+FLAGS.num_v+5551, :])
                features = sparse_to_tuple(features.tocoo())
                num_features = features[2][1]
                features_nonzero = features[1].shape[0]
                adj_norm = preprocess_graph(sp.vstack(sp.hstack(adj_citeulike[i:i+FLAGS.num_u, i:i+FLAGS.num_u],
                                                                adj_citeulike[i:i+FLAGS.num_u, 5551+j:j+FLAGS.num_v+5551]),
                                                      sp.hstack(adj_citeulike[5551+j:j+FLAGS.num_v+5551, i:i+FLAGS.num_u],
                                                                adj_citeulike[5551+j:j+FLAGS.num_v+5551, 5551+j:j+FLAGS.num_v+5551])))
                feed_dict = construct_feed_dict(adj_norm,
                                                tt[i:i+FLAGS.num_u,5551+j:j+FLAGS.num_v+5551],
                                                features, placeholders)
                C = np.empty((FLAGS.num_u, FLAGS.num_v))
                C.fill(FLAGS.b)
                C[tt[i:i+FLAGS.num_u,5551+j:j+FLAGS.num_v+5551]>0] = FLAGS.a

            if i+FLAGS.num_u >= 5551 & j+FLAGS.num_v < 16980:
                features = sp.vstack(sp.csr_matrix(features)[i:5551, :],
                                     sp.csr_matrix(features)[5551 + j:j + FLAGS.num_v + 5551, :])
                features = sparse_to_tuple(features.tocoo())
                num_features = features[2][1]
                features_nonzero = features[1].shape[0]
                adj_norm = preprocess_graph(sp.vstack(sp.hstack(adj_citeulike[i:5551, i:5551],
                                                                adj_citeulike[i:5551,
                                                                5551 + j:j + FLAGS.num_v + 5551]),
                                                      sp.hstack(adj_citeulike[5551 + j:j + FLAGS.num_v + 5551,
                                                                i:5551],
                                                                adj_citeulike[5551 + j:j + FLAGS.num_v + 5551,
                                                                5551 + j:j + FLAGS.num_v + 5551])))
                feed_dict = construct_feed_dict(adj_norm,
                                                tt[i:5551, 5551 + j:j + FLAGS.num_v + 5551],
                                                features, placeholders)
                C = np.empty((5551-i, FLAGS.num_v))
                C.fill(FLAGS.b)
                C[tt[i:5551, 5551 + j:j + FLAGS.num_v + 5551] > 0] = FLAGS.a

            if i + FLAGS.num_u < 5551 & j + FLAGS.num_v >= 16980:
                features = sp.vstack(sp.csr_matrix(features)[i:i + FLAGS.num_u, :],
                                     sp.csr_matrix(features)[5551 + j:, :])
                features = sparse_to_tuple(features.tocoo())
                num_features = features[2][1]
                features_nonzero = features[1].shape[0]
                adj_norm = preprocess_graph(sp.vstack(sp.hstack(adj_citeulike[i:i + FLAGS.num_u, i:i + FLAGS.num_u],
                                                                adj_citeulike[i:i + FLAGS.num_u,
                                                                5551 + j:]),
                                                      sp.hstack(adj_citeulike[5551 + j:,
                                                                i:i + FLAGS.num_u],
                                                                adj_citeulike[5551 + j:,
                                                                5551 + j:])))
                feed_dict = construct_feed_dict(adj_norm,
                                                tt[i:i + FLAGS.num_u, 5551 + j:],
                                                features, placeholders)
                C = np.empty((FLAGS.num_u, 16980-j))
                C.fill(FLAGS.b)
                C[tt[i:i + FLAGS.num_u, 5551 + j:j + FLAGS.num_v + 5551] > 0] = FLAGS.a

            if i + FLAGS.num_u >= 5551 & j + FLAGS.num_v >= 16980:
                features = sp.vstack(sp.csr_matrix(features)[i:5551, :],
                                     sp.csr_matrix(features)[5551 + j:, :])
                features = sparse_to_tuple(features.tocoo())
                num_features = features[2][1]
                features_nonzero = features[1].shape[0]
                adj_norm = preprocess_graph(sp.vstack(sp.hstack(adj_citeulike[i:5551, i:5551],
                                                                adj_citeulike[i:5551,
                                                                5551 + j:]),
                                                      sp.hstack(adj_citeulike[5551 + j:,
                                                                i:5551],
                                                                adj_citeulike[5551 + j:,
                                                                5551 + j:])))
                feed_dict = construct_feed_dict(adj_norm,
                                                tt[i:5551, 5551 + j:],
                                                features, placeholders)
                C = np.empty((5551-i, 16980-j))
                C.fill(FLAGS.b)
                C[tt[i:5551, 5551 + j:] > 0] = FLAGS.a

            feed_dict.update({placeholders['rate_a_b']: C})
            feed_dict.update({placeholders['num_features': num_features]})
            feed_dict.update({placeholders['features_nonzero': features_nonzero]})
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
            avg_cost = outs[1]
            j += FLAGS.num_v
            print("Iter:", '%04d' % (iter + 1), "train_loss=", "{:.5f}".format(avg_cost))
            iter += 1
        i += FLAGS.num_u
        t = time.time()
    # Construct feed dictionary
    #feed_dict = construct_feed_dict(adj_norm, tt, features, placeholders)
    #feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update



    #outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)


    #te = sess.run([model.embeddings, model.reconstructions], feed_dict = feed_dict)
    #ss = np.array(te[1])
    #print (ss.shape)
    #print(tf.shape(te[1]))

    # Compute average loss
    #avg_cost = outs[1]
    #avg_accuracy = outs[2]

    #roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    #val_roc_score.append(roc_curr)

    #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
    #     "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

#roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
#print('Test ROC score: ' + str(roc_score))
#print('Test AP score: ' + str(ap_score))
