from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Recommender_Decoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.mylayer1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)

        self.hidden1 = self.mylayer1(self.inputs)

        self.mylayer2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)

        self.embeddings = self.mylayer2(self.hidden1)
        self.W1 = self.mylayer1.get_weight()
        self.W2 = self.mylayer2.get_weight()
        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)


class RecommenderGCNModelAE(Model):
    def __init__(self, placeholders, input_dim_users, input_dim_items, **kwargs):
        super(RecommenderGCNModelAE, self).__init__(**kwargs)

        self.inputs_users = placeholders['features_user']
        self.input_dim_users = input_dim_users
        self.features_nonzero_users = placeholders['features_nonzero_user']
        self.adj_user = placeholders['adj_norm_user']

        self.inputs_items = placeholders['features_item']
        self.input_dim_items = input_dim_items
        self.features_nonzero_items = placeholders['features_nonzero_item']
        self.adj_item = placeholders['adj_norm_item']

        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1_user = GraphConvolutionSparse(input_dim=self.input_dim_items,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj_user,
                                              features_nonzero=self.features_nonzero_users,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs_users)

        self.embeddings_user = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj_user,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1_user)

        self.hidden1_item = GraphConvolutionSparse(input_dim=self.input_dim_users,
                                                   output_dim=FLAGS.hidden1,
                                                   adj=self.adj_item,
                                                   features_nonzero=self.features_nonzero_items,
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging)(self.inputs_items)

        self.embeddings_item = GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                adj=self.adj_item,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                logging=self.logging)(self.hidden1_item)

        self.reconstructions = Recommender_Decoder(
                                      act=lambda x: x,
                                      num_u = FLAGS.num_u,
                                      num_v = FLAGS.num_v,
                                      logging=self.logging)(tf.concat([self.embeddings_user, self.embeddings_item], 0))

class GCNModelAE_CITE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, num_u, num_v, **kwargs):
        super(GCNModelAE_CITE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.num_u = num_u
        self.num_v = num_v
        self.build()

    def _build(self):

        self.mylayer1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)
        self.hidden1 = self.mylayer1(self.inputs)

        self.mylayer2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)

        self.embeddings = self.mylayer2(self.hidden1)

        self.z_mean = self.embeddings

        self.W1 = self.mylayer1.get_weight()
        self.W2 = self.mylayer2.get_weight()

        self.reconstructions = Recommender_Decoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      num_u = self.num_u,
                                      num_v = self.num_v,
                                      logging=self.logging)(self.embeddings)
