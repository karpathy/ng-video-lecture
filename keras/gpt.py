"""
Keras version of the Transformer
"""

import math
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from typing import Callable, List

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


class TransformerModel:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout_rate, learning_rate, random_seed=1116):
        self.block_size = block_size
        self.vocab_size = vocab_size

        keras.utils.set_random_seed(random_seed)
        inputs = keras.Input((block_size,))
        outputs = TransformerLayer(vocab_size, n_embd, n_head, n_layer, dropout_rate)(inputs)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.summary()

    def estimate_loss(self, num_iters: int, data: Data) -> dict:
        res = dict()
        for split in [data.TRAIN_SPLIT, data.VAL_SPLIT]:
            loss = np.mean([self.model.evaluate(*data.fetch_batch(split), verbose=0)
                           for _ in range(num_iters)])
            res[split] = loss
            print(f'{split} loss {loss:.4f}')

        return res

    def generate_text(self, max_new_tokens: int, decoder: Callable) -> List[int]:
        res = []
        idx = [0] * self.block_size

        for _ in range(max_new_tokens):
            idx_cond = idx[-self.block_size:]  # crop idx to the last block_size tokens
            logits = self.model.predict(np.array([idx_cond]), verbose=0)
            logits = logits[0, -1, :]  # focus only on the last time step
            probs = softmax(logits, axis=-1)  # apply softmax to get probabilities
            idx_next = np.random.choice(range(self.vocab_size), 1, p=probs)[0]
            idx.append(idx_next)
            res.append(idx_next)
            print(decoder([idx_next]), end='')

        print()
        return res

    def train_on_batch(self, x, y, *args, **kwargs):
        return self.model.train_on_batch(x, y, *args, **kwargs)


# ---------- hyperparameters ----------
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout_rate = 0.2

# ---------- train ----------
data = Data(block_size, batch_size)
transformer = TransformerModel(data.vocab_size, n_embd, n_head, n_layer, block_size, dropout_rate, learning_rate)

for i in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0 or i == max_iters - 1:
        print(f'Step {i}')
        transformer.estimate_loss(eval_iters, data)

        print('Text generated:')
        transformer.generate_text(500, data.decoder)

    xb, yb = data.fetch_batch(Data.TRAIN_SPLIT)
    loss = transformer.train_on_batch(xb, yb)
