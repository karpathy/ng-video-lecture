#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTParams:
    def __init__(self):
        # hyperparameters
        self._batch_size = 64 # how many independent sequences will we process in parallel?
        self._block_size = 256 # what is the maximum context length for predictions?
        self._max_iters = 5000
        self._eval_interval = 500
        self._learning_rate = 3e-4
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._eval_iters = 200
        self._n_embd = 384
        self._n_head = 6
        self._n_layer = 6
        self._dropout = 0.2
        self._head_size =  self._n_embd // self._n_head

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def block_size(self):
        return self._block_size

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def eval_interval(self):
        return self._eval_interval

    @property
    def learning_rate(self):
        return self._learning_rate
    @property
    def device(self):
        return self._device

    @property
    def eval_iters(self):
        return self._eval_iters

    @property
    def n_embd(self):
        return self._n_embd

    @property
    def n_head(self):
        return self._n_head

    @property
    def n_layer(self):
        return self._n_layer

    @property
    def dropout(self):
        return self._dropout
    @property
    def head_size(self):
        return self._head_size

# data loading
def get_batch(split, gpt_params):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - gpt_params.block_size, (gpt_params.batch_size,))
    x = torch.stack([data[i:i+gpt_params.block_size] for i in ix])
    y = torch.stack([data[i+1:i+gpt_params.block_size+1] for i in ix])
    x, y = x.to(gpt_params.device), y.to(gpt_params.device)
    return x, y

@torch.no_grad()
def estimate_loss(gpt_params):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(gpt_params.eval_iters)
        for k in range(gpt_params.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, gpt_params):
        super().__init__()
        self.key = nn.Linear(gpt_params.n_embd, gpt_params.head_size, bias=False)
        self.query = nn.Linear(gpt_params.n_embd, gpt_params.head_size, bias=False)
        self.value = nn.Linear(gpt_params.n_embd, gpt_params.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(gpt_params.block_size, gpt_params.block_size)))

        self.dropout = nn.Dropout(gpt_params.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, gpt_params):
        super().__init__()
        self.heads = nn.ModuleList([Head(gpt_params) for _ in range(gpt_params.n_head)])
        self.proj = nn.Linear(gpt_params.head_size * gpt_params.n_head, gpt_params.n_embd)
        self.dropout = nn.Dropout(gpt_params.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, gpt_params):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gpt_params.n_embd, 4 * gpt_params.n_embd),
            nn.ReLU(),
            nn.Linear(4 * gpt_params.n_embd, gpt_params.n_embd),
            nn.Dropout(gpt_params.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, gpt_params):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(gpt_params)
        self.ffwd = FeedFoward(gpt_params)
        self.ln1 = nn.LayerNorm(gpt_params.n_embd)
        self.ln2 = nn.LayerNorm(gpt_params.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, gpt_params, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, gpt_params.n_embd)
        self.position_embedding_table = nn.Embedding(gpt_params.block_size, gpt_params.n_embd)
        self.blocks = nn.Sequential(*[Block(gpt_params) for _ in range(gpt_params.n_layer)])
        self.ln_f = nn.LayerNorm(gpt_params.n_embd) # final layer norm
        self.lm_head = nn.Linear(gpt_params.n_embd, vocab_size)
        self._gpt_params = gpt_params

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self._gpt_params.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, gpt_params):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -gpt_params.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def process_input_file(file_name, return_data=True):
        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    data = torch.tensor(encode(text), dtype=torch.long)
    if return_data:
        return encode, decode, vocab_size, data
    else:
        return encode, decode, vocab_size

def main():
    
    torch.manual_seed(1337)

    encode, decode, vocab_size, data = process_input_file('input.txt')


    
    # Train and test splits
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
            
    gpt_params = GPTParms()
    model = GPTLanguageModel(gpt_params, vocab_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
    
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save the checkpoint
            checkpoint = {
                'epoch': iter // eval_interval,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, 'checkpoint.pt')
    
    
        # sample a batch of data
        xb, yb = get_batch('train')
    
        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000, gpt_params=gpt_params)[0].tolist()))

if __name__ == '__main__':
    main()
