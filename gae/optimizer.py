import tensorflow as tf
import numpy as np
import time
flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class RecommenderOptimizerAE(object):
    def __init__(self, preds, labels, a, b, num_u, num_v, C, W1, W2, mylambda):
        #preds_sub = np.array(preds)
        #labels_sub = np.array(labels[0:5551,5551:22531])
        self.cost = 0
        #index_end = int(num_v+num_u)
        print("Recommenderopt %f" %time.time())
        #C = tf.fill([num_u,num_v], b)
        #index_bool = (labels[0:num_u,num_u:index_end] > 0)
        #tf.gather(C, index_bool) = a

        #E_matrix = np.array((C*((labels_sub - preds_sub)**2))/(2+0.0))
        #E_matrix = np.array((C * ((labels[0:5551,5551:22531]-preds) ** 2)) / (2 + 0.0))
        self.cost = tf.reduce_sum(((C * ((labels-preds) ** 2)) / (2 + 0.0))) + mylambda*tf.nn.l2_loss(W1) + mylambda*tf.nn.l2_loss(W2)
        print ("Cost finish: %f" %time.time())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        #self.grads_vars = self.optimizer.compute_gradients(self.cost)

        #self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
        #                                  tf.cast(labels_sub, tf.int32))
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class RecommenderOptimizerAE_one(object):
    def __init__(self, preds, labels, a, b, num_u, num_v, users, items, m_U, m_V):
        preds_sub = preds
        labels_sub = labels
        self.cost = 0

        ids = np.array([len(x) for x in users]) > 0
        u = m_U[ids]
        XX = np.dot(u.T, u) * b
        for j in xrange(num_v):
            user_ids = items[j]
            m = len(user_ids)
            if m > 0:
                A = np.copy(XX)
                A += np.dot(self.m_U[user_ids, :].T, self.m_U[user_ids, :]) * (a-b)
                B = np.copy(A)

                self.cost += 0.5 * m * a
                self.cost += -a * np.sum(np.dot(m_U[user_ids, :], m_V[j, :][:, np.newaxis]), axis=0)
                self.cost += 0.5 * m_V[j, :].dot(B).dot(m_V[j, :][:, np.newaxis])
            else:
                #not sure whether this term should be considered, cvae does not consider it
                self.cost += 0.5 * m_V[j, :].dot(XX).dot(m_V[j, :][:, np.newaxis])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        # self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(preds_sub, 0.5), tf.int32),
        #                                  tf.cast(labels_sub, tf.int32))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))