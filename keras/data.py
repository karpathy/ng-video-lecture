import numpy as np


class Data:
  TRAIN_SPLIT, VAL_SPLIT = 'train', 'val'

  def __init__(self, seq_len, batch_size, random_seed=0):
    """
    The Data class that streams data batches for the Keras model
    """

    np.random.seed(random_seed)
    self.seq_len = seq_len    # e.g. 8
    self.batch_size = batch_size

    # read it in to inspect it
    with open('../input.txt', 'r', encoding='utf-8') as f:
      text = f.read()

    # here are all the unique characters that occur in this text
    self.chars = sorted(list(set(text)))
    self.vocab_size = len(self.chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(self.chars)}
    itos = {i: ch for i, ch in enumerate(self.chars)}
    self.encoder = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    self.decoder = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    data = self.encoder(text)

    n = int(.9 * len(data))
    self.data = {Data.TRAIN_SPLIT: np.array(data[:n]), Data.VAL_SPLIT: np.array(data[n:])}
    print("length of dataset in characters: ", len(self.data))


  def fetch_batch(self, split):
    """
    Generate one batch of data
    """
    assert split in {Data.TRAIN_SPLIT, Data.VAL_SPLIT}
    d = self.data[split]
    ix = np.random.randint(0, len(d) - self.seq_len, (self.batch_size,))

    x = np.vstack([d[i: i + self.seq_len] for i in ix])
    y = np.stack([d[i + 1 : i + self.seq_len + 1] for i in ix])

    return x, y
