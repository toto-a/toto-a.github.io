---
usemathjax : true
title : Modern Transformers 
header-includes:
- \usepackage{annotate-equations}
- \usepackage{tikz} 
- jekyll-mermaid
- jekyll-graphviz
figureTemplate: '*$$figureTitle$$ $$i$$*$$titleDelim$$ $$t$$'
--- 

In this blog post, I will try to explain and show with code the basis of modern transformers (relative postion, absolute, learnt embeddings, rms norm, rope, kv cache, and more ..).

## Transformers Introduction 

In recent years, Transformer models have revolutionized natural language processing (NLP), with GPT (Generative Pre-trained Transformer) and other advanced language models leading the charge. This blog post explores the technical intricacies that make these models so powerful, from the self-attention mechanism to the architectural nuances that set them apart.

An example of the architecture of an transformer :
![GPT](/images/Transformer/GPT.png)
 _Figure 1 : Transformer (Decoder only) architecture_

## Embeddings
For position emebdings, there are different type : absolute, relative and sinusoid.

But first, why do we even need position embedding in transformer ?
Well this is a good question, in previous model such as LSTM, or RNN, we didn't have such consideration. 

Indeed, however the main caracteristics of transformers, the Multi-Head Self Attention and in particular Self Attention encodes no positional information which makes it permutation invariant. 


So to remedy to this problem, and introduce a notion of structure, we use positional encodings. 

Sinusoid position embeddings as described in the vanilla transformer paper [[1]](#references) :

Formally they can be written as :

$$ PE_{(i,2j)}= sin(\frac{i}{10000^\frac{2j}{d_{model}}}) $$
$$ PE_{(i,2j+1)}= cos(\frac{i}{10000^\frac{2j}{d_{model}}}) $$

where i is the position index and j the dimension index

```python

class PositionEmbeddings(nn.Module) : 
    def __init__(self,config) -> None:
        super().__init__()
        self.config=config
        P=torch.zeros(self.config.seq_len,self.config.hidden_size)
        positions=torch.arange(0,config.seq_len ,dtype=torch.float).unsqueeze(1)
        
        #For numeric stability
        div_term=torch.exp(-torch.arange(0,config.hidden_size,2)*math.log(10000)/config.hidden_size)

        ## For even position
        P[:,0::2]=torch.sin(positions*div_term)

        # For odd position
        P[:,1::2]=torch.cos(positions*div_term)
        P=P.unsqueeze(0)
        self.register_buffer("PE",P)
    
    def forward(self, x : torch.tensor) :
        B,T,D=x.shape[1]
        out=x+(self.P[:,:T,:]).requires_grad_(False)
        
        return out

```








## Attention 

Without KV cache : 

````python 
class CausalAttention(nn.Module) :
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config=config
        self.n_heads=config.n_heads
        self.head_size=config.hidden_size//self.n_heads
        self.scale=self.head_size**-0.5

        self.qkv=nn.Linear(config.hidden_size, config.hidden_size *3)
        self.o=nn.Linear(config.hidden_size, config.hidden_size)

        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len,config.seq_len)).view(1,1,))

    def forward(self,x) :

        B,T,D=x.shape
        mixed_qkv=(self.qkv(x)
                            .view(3,B,T,self.head_size,self.n_heads)
                            .permute(0,1,4,2,3)
        )

        q,k,v=mixed_qkv[0],mixed_qkv[1],mixed_qkv[2]
        attn_scores=torch.matmul(q,k.transpose(-2,-1))*self.scale 
        attn_scores=attn_scores.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))

        attn_scores=F.softmax(attn_scores,dim=-1)
        context=attn_scores@v

        context=rearrange(context, 'b h t d->b t (h d)')

        return context
````
With KV cache :

![KV cache](/images/Transformer/KV.gif)

```python
class CausalAttentionKVCache(nn.Module):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config=config
        self.n_heads=config.n_heads
        self.head_size=config.hidden_size//self.n_heads
        self.scale=self.head_size**-0.5

        self.qkv=nn.Linear(config.hidden_size, config.hidden_size *3)
        self.o=nn.Linear(config.hidden_size, config.hidden_size)

        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len,config.seq_len)).view(1,1,))
        self.cache_k=None
        self.cache_v=None

    
    def get_cache(self, x: torch.tensor) :

        if self.cache_k is None :
            self.cache_k=torch.empty(
                self.config.batch_size,
                self.config.seq_len,
                self.n_heads,
                self.head_size,
                device=x.device

            )
        
        if self.cache_v is None : 
            self.cache_v=torch.empty(
                 self.config.batch_size,
                self.config.seq_len,
                self.n_heads,
                self.head_size,
                device=x.device
            )
        
        return self.cache_k,self.cache_v


    def forward(self,x) :

        B,T,D=x.shape
        
        cache_k,cache_v=self.get_cache(x)

        mixed_qkv=(self.qkv(x)
                            .view(3,B,T,self.head_size,self.n_heads)
                            .permute(0,1,4,2,3)
        )
        q,k,v=mixed_qkv[0],mixed_qkv[1],mixed_qkv[2]

        ## Create Positions 
        positions=torch.arange(0,self.config.max_seq_len)[None,:,None,None].repeat(
            B,1,self.n_heads,self.head_size
        )

        ##Update cache, Replace with the new entry
        cache_k[:B].scatter_(dim=1, index=positions, src=k)
        cache_v[:B].scatter_(dim=1, index=positions, src=v)

        ## Perform attention with the current cache
        k,v=cache_k.transpose(1,2), cache_v.transpose(1,2)

        attn_scores=torch.matmul(q,k.transpose(-2,-1))*self.scale
        attn_scores=attn_scores.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        attn_scores=F.softmax(attn_scores,dim=-1)
        context=attn_scores@v

        context=rearrange(context, 'b h t d->b t (h d)')

        return context
```

## MLP

Once attention has moved to relevant info in the residual stream, we will use MLPs to do some reasoning on those informations

With SILU : 

```python 
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
   

```

## Normalization 

```python 
class RMSNorm(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```



## References

[1] VASWANI, Ashish, SHAZEER, Noam, PARMAR, Niki, et al. Attention is all you need. Advances in neural information processing systems, 2017, vol. 30.

[2] Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 127063.
