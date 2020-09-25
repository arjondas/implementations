import torch
import torch.nn as nn

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    '''
    embed_size: Size of input embedding features
    heads: Number of heads to use
    '''
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (self.head_dim * self.heads == embed_size), 'Embed size needs to multiple of heads'

    # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)


    self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
    self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
    self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)

    self.fc_out = nn.Linear(self.embed_size, self.embed_size)

  def forward(self, values, keys, queries, mask):
    batch_size = queries.shape[0]    ## Batch size
    value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

    keys = self.keys(keys)
    values = self.values(values)
    queries = self.queries(queries)

    ## Split embeddings into head pieces
    ## e.g. embed_size: 256, heads: 8, then (batch_size, k/v/q_len, 256) => (batch, k/v/q_len, 8, 32)
    keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
    values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
    queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)

    # energy shape: (batch_size, heads, query_len, key_len)
    # rough intuition: take the query(input) and find the most similar keys
    # TODO: Convert to batch multiplication: torch.bmm(...)
    energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

    if mask is not None:
      energy = energy.masked_fill(mask==0, float('-1e20'))
    
    # Making attention scores normalize to one across source input
    attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

    # attention shape: (batch_size, heads, query_len, key_len)
    # value shape: (batch_size, value_len, heads, head_dim)
    # basically going to multipy the attention with the value
    out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(
      batch_size, query_len, self.heads*self.head_dim
    )

    out = self.fc_out(out)
    return out

