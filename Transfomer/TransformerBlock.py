import torch
import torch.nn as nn
from SelfAttention import SelfAttention

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion*embed_size),
      nn.ReLU(),
      nn.Linear(embed_size*forward_expansion, embed_size)
    )
    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query, mask):
    attention = self.attention(value, key, query, mask)
    x = self.dropout(self.norm1(attention + query))     ## skip ahead conncetion
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))     ## skip ahead connection
    return out
