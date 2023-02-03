"""
Keras version of the Transformer
"""

import math
import numpy as np
import tensorflow as tf
keras = tf.keras        # silly imports to work around PyCharm inspections
layers = keras.layers
from keras import backend as K

from data import Data


class Head(layers.Layer):
    def __init__(self, head_size, dropout_rate):
        super().__init__()
        self.head_size = head_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.key = layers.Dense(self.head_size, activation=None, use_bias=False)
        self.query = layers.Dense(self.head_size, activation=None, use_bias=False)
        self.value = layers.Dense(self.head_size, activation=None, use_bias=False)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.mask = tf.constant(np.tril(np.ones((input_shape[1], input_shape[1]))))

    def call(self, x, *args, **kwargs):
        k = self.key(x)     # (B,T,C)
        q = self.query(x)   # (B,T,C)

        # compute attention scores ("affinities")
        k_transposed = K.permute_dimensions(k, (0, 2, 1))   # B, C, T
        wei = tf.matmul(q, k_transposed)    # B, T, T
        wei /= math.sqrt(x.shape[-1])     # scale
        wei = layers.Softmax(axis=-1)(wei, self.mask)   # softmax while making the upper-triangle all 0
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)   # B, T, C
        out = tf.matmul(wei, v)      # (B, T, T) @ (B, T, C) -> (B, T, C)

        assert out.shape[1] == x.shape[1] and out.shape[2] == self.head_size

        return out

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_size, n_embd, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.heads = [Head(self.head_size, self.dropout_rate) for _ in range(self.num_heads)]
        self.proj = layers.Dense(self.n_embd, activation=None)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, *args, **kwargs):
        out = layers.Concatenate(axis=-1)([h(x) for h in self.heads])
        out = self.dropout(self.proj(out))

        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd

        return out


class FeedForward(layers.Layer):
    def __init__(self, n_embd: int, dropout_rate: float):
        super().__init__()

        self.n_embd = n_embd
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.net = keras.Sequential([
            layers.Conv1D(filters=4, kernel_size=1, activation='relu'),
            layers.Conv1D(filters=self.n_embd, kernel_size=1),
            layers.Dropout(self.dropout_rate)
        ])

    def call(self, x, *args, **kwargs):
        out = self.net(x)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd

        return out


class Block(layers.Layer):
    def __init__(self, n_embd: int, n_head: int, dropout_rate: float):
        assert n_embd % n_head == 0
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.sa = MultiHeadAttention(self.n_head, self.head_size, self.n_embd, self.dropout_rate)
        self.ffwd = FeedForward(self.n_embd, self.dropout_rate)
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, *args, **kwargs):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerLayer(layers.Layer):
    def __init__(self, vocab_size, n_embd, n_head, n_block, dropout_rate):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.n_block = n_block

    def build(self, input_shape):
        self.block_size = input_shape[1]

        self.token_embedding_table = layers.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = layers.Embedding(self.block_size, self.n_embd)
        self.blocks = keras.Sequential([Block(self.n_embd, self.n_head, self.dropout_rate) for _ in range(self.n_block)])
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(self.vocab_size, activation=None)

    def call(self, idx, *args, **kwargs):
        # idx is of shape (B, T, C)

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(0, self.block_size)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits


keras.utils.set_random_seed(1116)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

train_data = Data(block_size, batch_size, split='train', data_pts_per_epoch=max_iters)
val_data = Data(block_size, batch_size, split='test', data_pts_per_epoch=eval_iters)

inputs = keras.Input((block_size,))
outputs = TransformerLayer(train_data.vocab_size, n_embd, n_head, n_layer, dropout)(inputs)
m = keras.Model(inputs, outputs)
m.compile(optimizer=keras.optimizers.Adam(learning_rate),
          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
m.fit(train_data, epochs=1, validation_freq=eval_interval)
m.train_on_batch(train_data)