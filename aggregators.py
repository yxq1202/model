import warnings

warnings.filterwarnings('ignore')
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [4096,1,1,8]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
            # [4096,1,5,8]
            user_relation_cat = tf.concat([user_embeddings, neighbor_relations], axis=2)
            # [4096,1,5]
            user_relation_cat = tf.reduce_sum(user_relation_cat, axis=3)
            layer1 = tf.layers.Dense(self.dim, activation=tf.nn.tanh, name='att_layer1')
            layer2 = tf.layers.Dense(4, activation=tf.nn.leaky_relu, name='att_layer2')
            layer1_output = layer1(user_relation_cat)
            layer2_output = layer2(layer1_output)
            # tensor [4096,1,4]
            user_relation_attention = tf.exp(layer2_output)
            user_relation_attention_normalized = tf.nn.softmax(user_relation_attention, axis=-1)
            # [4096,1,4,1]
            user_relation_attention_normalized = tf.expand_dims(user_relation_attention_normalized, axis=-1)
            # Perform neighbors aggregation,[4096,-1,8]
            neighbors_aggregated = tf.reduce_mean(user_relation_attention_normalized * neighbor_vectors, axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)


        # if not avg:
        #     user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
        #     user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        #     user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)
        #     user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)
        #     neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        # else:
        #     # [batch_size, -1, dim]
        #     neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        # variable_scope：管理同名的参数
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.compat.v1.keras.initializers.glorot_normal(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        # [65536,1,32]
        # [65536,4,32]
        # 得到了所有邻居对自己的补充信息
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        # [-1, dim]
        # (65536, 1, 32)+[65536,1,32]===>[65536,1,32]===>[65536,32]
        # (65536, 4, 32)+[65536,4,32]===>[65536,4,32]===>[262144,32]
        # 把邻居信息和自身信息用sum聚合器聚合
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1 - self.dropout)
        # output = tf.matmul(output, self.weights) + self.bias
        # [batch_size, -1, dim]
        # [65536,1,32]
        # [65536,4,32]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        return output
