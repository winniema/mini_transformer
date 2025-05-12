import torch.nn as nn 
import torch
from torch.nn import functional as F

"""
A mini transformer with two repeated blocks of masked multi-head attention and feed forward layers. 

Implemented based on Andrej Karpathy's "Let's Build GPT" video (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5868s), which 
follows the "Attention is All You Need" paper (https://arxiv.org/pdf/1706.03762).
Slightly modified to be able to run on a M3 Macbook Air 
"""
class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, block_size, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """ Input (B, T, C), Output: (B, T, hs) """
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention with a linear projection at the end """

    def __init__(self, block_size, num_heads, n_embd):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(block_size, head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ Two linear transformations with a ReLU activation in between """

    def __init__(self, n_embd):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
      return self.net(x)

class Block(nn.Module):
    """ 
    A repeatable block within a transformer, consistenting of layer norm to masked multi-headed attention, 
    to another layer norm and a final feed forward layer.
    Residual pathways are added to short circuit computation in 2 places: 1) the multi-headed attention heads, and 2) the feed
    forward layer. The motivation is detailed in the paper "Deep Residual Learning for Image Recognition"
    (https://arxiv.org/pdf/1512.03385).
    """

    def __init__(self, block_size, num_heads, n_embd):
        super().__init__()
        self.sa_heads = MultiHeadAttention(block_size, num_heads, n_embd)
        self.feed_forward = FeedForward(n_embd)
        self.lnorm1 = nn.LayerNorm(n_embd)
        self.lnorm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.lnorm1(x))
        out = x + self.feed_forward(self.lnorm2(x))
        return out
    
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        num_heads = 4 # Select num_heads such that attention blocks can concat back to n_embd size
        
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        
        self.blocks = nn.Sequential(
            Block(self.block_size, num_heads, n_embd),
            Block(self.block_size, num_heads, n_embd),
        )
        self.lnorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): 
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, n_embed)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.lnorm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            idx_blocked = idx[:, -self.block_size:]
            logits, loss = self(idx_blocked)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx