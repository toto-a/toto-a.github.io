---
usemathjax : true
title : Modern Transformers 
header-includes:
   - \usepackage{annotate-equations}

figureTemplate: '*$$figureTitle$$ $$i$$*$$titleDelim$$ $$t$$'
--- 

In this blog post, I will try to explain and show with code the basis of modern transformers (relative postion, absolute, learnt embeddings, rms norm, rope, kv cache, and more ..)

## Transformers Introduction 

## Embeddings
For position emebdings, there are different type : absolute, relative and learnt.
Absolute position embeddings as described in the paper [[1]](#references)




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


## MLP

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