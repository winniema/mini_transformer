import torch
from MiniTransformer import *

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/Users/winniema/IntroModels/Bigram/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Here are all the unique characters that occur in this text
# Each character is treated as a token
unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
print(f'{vocab_size=}')

# Define char encode (str to int) and decode (int to str) functions
stoi = {ch:i for i,ch in enumerate(unique_chars)}
encode = lambda s: [stoi[c] for c in s]
itos = {i:ch for i,ch in enumerate(unique_chars)}
decode = lambda nums: "".join([itos[num] for num in nums])

# Our entire dataset, as a 1-D tensor of shape [1115394]
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into train and validation sets. First 90% will be
# training data, the rest will be validation 
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    We select batch_size number of random batches as input x, and targets y to feed into the model. 
    This is so the model can process batch_size number of input, target pairs simultaneously. 
    These batches are selected at random for each training step. Given enough training steps (cycles
    through the model), we should get fairly representative coverage  of the overall training/validation
    data. 
    """
    data = train_data if split == 'train' else val_data

    # Get batch_size number of random ints between 0 and len(data)-block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    For a snapshot of the model during training, estimate the loss across eval_iters number of batches.
    @torch.no_grad(), m.eval(), m.train() tells the program that we are only doing inference and no 
    back propagation will be called for this calculation. 
    """
    m.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# Block size is the maximum context window
# batch_size affects how many examples in parallel the model will see before each
# update via back prop. Batch size that's too small will be difficult for the model
# to converge (i.e small sample is not representative of the overall data), too big 
# will lead to overfitting (i.e. imagine sending all training data to the model 
# every time) 
# block_size is the maximum context window (i.e. history the model can see)
"""
Set model parameters 

batch_size
* number of examples in parallel that the model will see before each update via back prop
* batch_size that's too small will be difficult for the model to converge
    (i.e small sample is not representative of the overall data), but batch_size that's 
    too big will lead to overfitting (i.e. imagine sending all training data to the model every time)
* the batch dimension (B)

block_size
* the maximum context window (i.e. the history the model can see to determine each target)
* the time dimension (T)

n_embd
* each token is translated (embeded) into a 1D vector of length n_embd
* the channel dimension (C)
"""
torch.manual_seed(1337)
batch_size = 32 
block_size = 8
n_embd = 32

# Initialize the model
m = MiniTransformer(vocab_size, n_embd, block_size)

"""
Set training parameters

learning_rate
* step size for the optimizer that implements back prop. Adjusting this rate affects which model
    minimum you find and how quickly 

max_iters
* training iterations for the model

eval_iters
* this is used in two ways: 1) as number of training iterations that runs before we evaluate the model,
    and 2) the number of iterations used to evaluate the model's average loss
"""
learning_rate = 1e-3
max_iters = 5001
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
eval_iters = 250

for iter in range(max_iters): 
    if iter % eval_iters == 0:
        out = estimate_loss()
        print(f"step {iter}: train loss {out['train']:.4f}, val loss {out['val']:.4f}")

    xb, yb = get_batch('train') # (B,T)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model and decode to alphabet
zeros = torch.zeros((1, 1), dtype=torch.long)
output = m.generate(idx = zeros, max_new_tokens=1000)
out_str = decode(output[0].tolist())
print(out_str)
