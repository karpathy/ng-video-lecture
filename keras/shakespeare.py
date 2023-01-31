import numpy as np
import tensorflow as tf
keras = tf.keras        # silly imports to work around PyCharm inspections
layers = keras.layers
from transformer import Transformer, generate_tokens


class ShakespeareData(tf.keras.utils.Sequence):

  def __init__(self, seq_len, batch_size, split: str, random_seed=0, data_pts_per_epoch=None):
    np.random.seed(random_seed)
    assert split in {'train', 'test'}
    self.seq_len = seq_len    # e.g. 8
    self.batch_size = batch_size
    self.split = split

    # read it in to inspect it
    with open('./data/shakespeare.txt', 'r', encoding='utf-8') as f:
      text = f.read()

    # here are all the unique characters that occur in this text
    self.chars = sorted(list(set(text)))
    self.vocab_size = len(self.chars)
    print(''.join(self.chars))
    print(self.vocab_size)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(self.chars)}
    itos = {i: ch for i, ch in enumerate(self.chars)}
    self.encoder = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    self.decoder = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    data = self.encoder(text)

    n = int(.9 * len(data))
    self.data = np.array(data[:n] if split == 'train' else data[n:])
    print("length of dataset in characters: ", len(self.data))

    self.batches_per_epoch = (data_pts_per_epoch or len(self.data)) // self.batch_size


  def __getitem__(self, index):
    """
    Generate one batch of data
    Does not use index. Randomly selects.
    """
      
    ix = np.random.randint(0, len(self.data) - self.seq_len, (self.batch_size,))

    x = np.vstack([self.data[i: i + self.seq_len] for i in ix])
    y = np.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])

    return x, y

  def __len__(self):
    return self.batches_per_epoch

if __name__ == '__main__':
    data = ShakespeareData(seq_len=20, batch_size=32, split='train')
    batches = [data.__getitem__(0) for _ in range(10)]

    seq_len = 12
    batch_size = 32
    trainData = ShakespeareData(seq_len, batch_size, 'train', data_pts_per_epoch=5000)
    testData = ShakespeareData(seq_len, batch_size, 'test', data_pts_per_epoch=2000)

    inputs = keras.Input(shape=(seq_len,))

    m = keras.Model(inputs=inputs,
                    outputs=Transformer(head_size=8, num_heads=6, seq_len=seq_len,
                                        vocab_size=trainData.vocab_size, num_blocks=3, dropout=0.25)(inputs)
                    )

    m.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=5e-4),
              metrics=[])
    m.summary()

    n_epochs = 10

    for i in range(5):
      # generate some text
      print('-' * 10)
      print(f'Generated text after {i * n_epochs} epochs:')
      print(trainData.decoder(generate_tokens(m, trainData.vocab_size, seq_len, 0, 500)))

      # train
      m.fit(trainData, epochs=n_epochs, validation_data=testData)
