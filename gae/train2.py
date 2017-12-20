from __future__ import division
from __future__ import print_function

import time
import os
import scipy.io

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE, RecommenderOptimizerAE
from input_data import load_data, load_data_from_citeulike, load_data_from_sub_citeulike
from model import GCNModelAE, GCNModelVAE, RecommenderGCNModelAE, GCNModelAE_CITE
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
flags.DEFINE_float('a',1.0, 'rated')
flags.DEFINE_float('b',0.01, 'not rated')
flags.DEFINE_integer('num_u', 5551, 'number of users')              #     5551
flags.DEFINE_integer('num_v', 16980, 'number of items')             #     16980
flags.DEFINE_float('regularization',0.01, 'regularization')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
#adj, features = load_data(dataset_str)
adj_citeulike = load_data_from_sub_citeulike()

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj_citeulike
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#adj = adj_train

#print(adj)
if FLAGS.features == 0:
    features = sp.identity(adj_citeulike.shape[1])  # featureless features = sp.identity(adj.shape[0])

if FLAGS.features == 1:
    variables = scipy.io.loadmat("mult_nor.mat")
    content = np.array(variables['X'])
    mZero1 = np.zeros((5551, 8000))
    mZero2 = np.zeros((16980, 5551))
    Id = np.identity(5551)
    temp1 = np.hstack((Id, mZero1))
    temp2 = np.hstack((mZero2, content))
    features = sp.csr_matrix(np.vstack((temp1, temp2)))

# Some preprocessing
adj_norm = preprocess_graph(adj_citeulike)        #some problem, some rows are zeros totally

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'rate_a_b': tf.placeholder(tf.float32)
}

num_nodes = adj_citeulike.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]
print (num_features)

# Create model
model = None
if model_str == 'gcn_ae':
    pass
    #model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    pass
    #model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'recommender_ae':
    model = GCNModelAE_CITE(placeholders, num_features, features_nonzero, FLAGS.num_u, FLAGS.num_v)

#pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'recommender_ae':
        opt = RecommenderOptimizerAE(preds=model.reconstructions,
                                     labels=placeholders['adj_orig'],
                                     a=FLAGS.a, b=FLAGS.b, num_u=FLAGS.num_u, num_v=FLAGS.num_v,
                                     C=placeholders['rate_a_b'], W1=model.W1, W2=model.W2, mylambda = FLAGS.regularization)
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


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in xrange(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in xrange(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_precision += len(act_set & pred_set) / float(topk)
            true_users += 1

    return sum_precision / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def read_user(f_in='cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.zeros((num_u,num_v))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R


def load_rating(path):
    arr = []
    for line in open(path):
        a = line.strip().split()
        if a[0] == 0:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr


def cal_precision(R):
    R_true = load_rating('80_20/cf-test-1-users.dat')
    R_train = read_user('80_20/cf-train-1-users.dat', num_u = 5551, num_v=16980)
    R = R * (R_train == 0)
    PP = (-R).argsort()
    PP = PP.tolist()

    r5 = recall_at_k(R_true, PP, 5)
    r10 = recall_at_k(R_true, PP, 10)
    r15 = recall_at_k(R_true, PP, 15)
    r20 = recall_at_k(R_true, PP, 20)
    r50 = recall_at_k(R_true, PP, 50)

    print('Recall@5: %.10f' % r5)
    print('Recall@10: %.10f' % r10)
    print('Recall@15: %.10f' % r15)
    print('Recall@20: %.10f' % r20)
    print('Recall@50: %.10f' % r50)

    p5 = precision_at_k(R_true, PP, 5)
    p10 = precision_at_k(R_true, PP, 10)
    p15 = precision_at_k(R_true, PP, 15)
    p20 = precision_at_k(R_true, PP, 20)
    p50 = precision_at_k(R_true, PP, 50)

    print('Precision@5: %.10f' % p5)
    print('Precision@10: %.10f' % p10)
    print('Precision@15: %.10f' % p15)
    print('Precision@20: %.10f' % p20)
    print('Precision@50: %.10f' % p50)

    m5 = mapk(R_true, PP, 5)
    m10 = mapk(R_true, PP, 10)
    m15 = mapk(R_true, PP, 15)
    m20 = mapk(R_true, PP, 20)
    m50 = mapk(R_true, PP, 50)

    print('MAP@5: %.10f' % m5)
    print('MAP@10: %.10f' % m10)
    print('MAP@15: %.10f' % m15)
    print('MAP@20: %.10f' % m20)
    print('MAP@50: %.10f' % m50)
    '''
    num_u = R.shape[0]
    print(num_u)
    num_hit = 0
    t = 0
    #fp = open(dir_save+'/rec-list.dat','w')
    for i in range(num_u):
        if i!=0 and i%100==0:
            print ('Iter '+str(i))

        l_score = np.ravel(R[i,:]).tolist()
        ll_score = R[i,:].tolist()
        #print '*************************'
        #print l_score []
        #print ll_score [[]]

        pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)

        #print pl


        l_rec = list(zip(*pl)[0])[:cut]
        #print l_rec

        s_rec = set(l_rec)
        s_true = set(np.ravel(np.where(R_true[i,:]>0)[1]))
        #print s_true

        if len(s_true) > 0:
            cnt_hit = len(s_rec.intersection(s_true))


            num_hit += float(cnt_hit)/float(len(s_true))
            t += 1
            #print num_hit
            #fp.write('%d:' % i)
            #fp.write('%d:' % cnt_hit)
            #fp.write(' '.join(map(str,l_rec)))
            #fp.write('\n')
    #fp.close()
    print ('Precision: %.3f' % (float(num_hit)/float(t)))
    '''


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_citeulike
adj_label = sparse_to_tuple(adj_label)
tt = adj_orig.tocoo().todense()
tt = np.array(tt)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, tt[0:FLAGS.num_u, FLAGS.num_u:FLAGS.num_u+FLAGS.num_v], features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    C = np.empty((FLAGS.num_u, FLAGS.num_v))
    C.fill(FLAGS.b)
    print(C.shape)
    C[tt[0:FLAGS.num_u, FLAGS.num_u:FLAGS.num_v+FLAGS.num_u] > 0] = FLAGS.a
    feed_dict.update({placeholders['rate_a_b']: C})

    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, model.reconstructions], feed_dict=feed_dict)

    #te = sess.run([model.embeddings, model.reconstructions], feed_dict = feed_dict)
    #ss = np.array(te[1])
    #print (ss.shape)
    #print(tf.shape(te[1]))

    # Compute average loss
    avg_cost = outs[1]
    R = outs[2]
    #avg_accuracy = outs[2]

    #roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    #val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "time=", "{:.5f}".format(time.time() - t))
    if (epoch+1) % 100 == 0:
        cal_precision(4, 50, R)

print("Optimization Finished!")

#roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
#print('Test ROC score: ' + str(roc_score))
#print('Test AP score: ' + str(ap_score))
