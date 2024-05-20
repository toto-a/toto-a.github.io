---
usemathjax : true
---


## Introduction to diffusion 

In my first blog post, I will try to explain all the concepts regarding diffusion, vae and ae. I will assume that you are a little familiar with ae and vae because I won't go into a lot of details about them.

### What are AE (Auto Encoders) 

Given an input $$x\in \mathbb{R}^D$$, the encoder $$D_{\theta}$$ will be used to generate an a latent variable z, to be passed to the decoder $$E_{\phi}$$, as shown in the image below  :

![_config.yml]({{ site.baseurl }}/images/ae.png)

As the goal is just to rescontruct the input, we want to learn the mapping 
$$\hat{x}=E_{\phi}(D_{\theta}(x)) \simeq x$$ 

We want to find the ideal network that will minimize the loss $$ L=\|\hat{x} - x\|^2$$ 
We will need to find 
$$ \min\limits_{\theta,\phi} L$$

The problem with autoencoders are that they are not a generative model, as it does not define a distribution. The biggest problem lies with its latent representations as it is deterministic, for same input it always generate same output.

### VAE (Variational autoencoders)