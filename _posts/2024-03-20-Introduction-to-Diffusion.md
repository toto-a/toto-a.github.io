---
usemathjax : true
title : Introduction to diffusion 
---



In my first blog post, I will cover the key concepts of diffusion, VAE, and AE. I'll assume you have some basic knowledge of AE and VAE, so I won't go into too much details into those topics.


### What are AE (Auto Encoders) 

Given an input $$x\in \mathbb{R}^D$$, the encoder $$D_{\theta}$$ will be used to generate an a latent variable z, to be passed to the decoder $$E_{\phi}$$, as shown in the image below  :

![_config.yml]({{ site.baseurl }}/images/ae.png)

As the goal is just to rescontruct the input, we want to learn the mapping 
$$\hat{x}=E_{\phi}(D_{\theta}(x)) \simeq x$$ 

We want to find the ideal network that will minimize the loss $$ L=\|\hat{x} - x\|^2$$ . 
We will need to find 
$$ \min\limits_{\theta,\phi} L$$

The problem with autoencoders are that they are not a generative model, as it does not define a distribution over the dataset. The biggest problem lies with its latent representations as it is deterministic, for same input it will always generate same output. 

It's like saying that each point in the latent space corresponds to a unique image.w

### VAE (Variational autoencoders)

So, if we want to train a generative model, we will want maximize the likelihood of the observed data (think of the term of text-modelling and llm). But we also want the latent space to be general enough to be able to give us a strong representations of the obserseved data.

Now let's consider some distributions : the one over the observation data, $$p(x)$$
* $$p(z)$$ 
the distribution of the latent variable and for the sake of simplicity let's suppose it is a unit-variance gaussian 
$$p(z)=\mathcal{N}(0,\,1)$$

* $$ p(x|z) $$ 
the probability distribution of the decoder (posterior probability of getting x given z)

With that done, we can define the likelihood of our distribution  : 

$$ p(x)= \displaystyle \int p(z)p(x,z) \, \mathrm{d}x $$

However, we don't have access to the joint distribution $$p(x,z)$$, so we can not really compute this likelihood in our case.

#### Evidence Lower Bound

So,it seems that to find both the encoder and the decoder we will need to estimate them. 

Consider the following distribution: 
* $$ q_{\phi}(z|x)$$ 
the posterior distrubtion and the estimation of 
$$p(z|x)$$

* $$ p_{\theta}(x|z)$$ 
an estimate for 
$$p(x|z)$$

We will assume that all distributions are gaussians. Why Gaussian ? it is because we want to impose 

As resumed by the figure below

![_config.yml]({{ site.baseurl }}/images/vaeb.png)

