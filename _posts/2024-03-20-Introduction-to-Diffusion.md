---
usemathjax : true
title : Introduction to diffusion 
header-includes:
   - \usepackage{annotate-equations}
---



In my first blog post, I will cover the key concepts of diffusion, VAE, and AE. I'll assume you have some basic knowledge of AE and VAE, so I won't go into too much details into those topics.


## What are AE (Auto Encoders) 

Given an input $$x\in \mathbb{R}^D$$, the encoder $$D_{\theta}$$ will be used to generate an a latent variable z, to be passed to the decoder $$E_{\phi}$$, as shown in the image below  :


{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/ae.png)
{: refdef}
{:refdef: style="text-align: center;"}
<figcaption> Figure 1 : Principle of an Auto-Encoders</figcaption>
{: refdef}

As the goal is just to rescontruct the input, we want to learn the mapping 
$$\hat{x}=E_{\phi}(D_{\theta}(x)) \simeq x$$ 

We want to find the ideal network that will minimize the loss $$ L=\|\hat{x} - x\|^2$$ . 
We will need to find 
$$ \arg \min\limits_{\theta,\phi} L$$

The problem with autoencoders are that they are not a generative model, as it does not define a distribution over the dataset. The biggest problem lies with its latent representations as it is deterministic, for same input it will always generate same output. 

It's like saying that each point in the latent space corresponds to a unique image.

## VAE (Variational autoencoders)

So, if we want to train a generative model, we will want maximize the likelihood of the observed data (think of the term of text-modelling and llm). But we also want the latent space to be general enough to be able to give us a strong representations of the obserseved data.

Now let's consider some distributions : the one over the observation data, our prior :

* $$p(x)$$

* $$p(z)$$ 
the distribution of the latent variable and for the sake of simplicity let's suppose it is a unit-variance gaussian 
$$p(z)=\mathcal{N}(0,\,1)$$

* $$p(z|x)$$ 
that describes the distribution of the encoded variable given the decoded one 

* $$ p(x|z) $$ 
that describes the distribution of the decoded variable given the encoded one and our likelihood.

With that done, let's consider this integral : 

$$ p(x)= \displaystyle \int p(x|z)p(z) \, \mathrm{d}z $$ 
Also called the _marginal likelihood_ or the _model evidence_. 

By Bayes theorem we have that :

$$ p(z|x)= \dfrac{p(x|z)p(z)}{\displaystyle \int p(x|z)p(z) \, \mathrm{d}z }$$

In theory, as we know $$p(z)$$ 
and $$p(x|z)$$ 
(a gaussian with a deterministic mean and variance),so we know the numerator. 

Now for the denominator, although in lower dimension we can compute each term. However in higher dimension, it becomes quickly intractable.
> Consider for example the case where $$x =x_{1:n} $$ 
>and $$z=z_{1:m}$$ 
>with $$n,m \in \mathbb{N^{*}}$$
>and $$ m<n $$
> The integral becomes : $$ p(x_i)= \displaystyle \int \sum_{z_{j}} p(z_{j})\prod_{i=1}^{n} p(x_i|z_j) \, \mathrm{d}z_j $$


### *Evidence Lower Bound*

So, it seems that to find both the encoder and the decoder we will need to approximate them. 

Consider the following distribution: 
* $$ q_{\phi}(z|x)$$ 
the posterior distrubtion which approximate
$$p(z|x)$$

* $$ p_{\theta}(x|z)$$ 
an estimate for 
$$p(x|z)$$

We will assume that all distributions are gaussians. Why Gaussian ? it is because we want to impose some conditions to be able to derive some general structure from it. 

As resumed by the figure below

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/vae_base.png)
{: refdef}
{:refdef: style="text-align: center;"}
<figcaption> Figure 2 : Principle of VAE</figcaption>
{: refdef}

Remember our integral :
$$p(x)= \displaystyle \int p(x,z) \, \mathrm{d}x $$ 

It now becomes :

$$p_{\theta}(x)= \displaystyle \int p_{\theta}(x,z) \, \mathrm{d}x $$

Then using our proxy 
$$q_{\phi}(z|x) $$ 
we have that : 

$$p_{\theta}(x)= \mathop{\mathbb{E_{q_{\phi}(z|x)}}}\Big[ \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)}\Big]$$


With Jensen inequality, we obtain : 

$$log(p_{\theta}(x)) \ge \mathop{\mathbb{E_{q_{\phi}(z|x)}}}\Big[\log  \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)}\Big]$$


Let's denote $$\boxed{ELBO(x) =\mathbb{E_{q_{\phi}(z|x)}} \Big[ \log \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)} \Big]} \ (1) $$ 
We have shown that ELBO is a valid lower bound on the log likelihood  of the data. Now let's look more closely at what is this ELBO. 

By Bayes theorem, we can develop our ELBO to look like : 
$$ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}} \Big[ \log \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)} \Big]$$
$$ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}}\Big[ \log p_{\theta}(x|z) \Big]+
 \mathbb{E_{q_{\phi}(z|x)}}\Big[\log\dfrac{p(z)}{q_{\phi}(z|x)}\Big]$$

Finally :
$$ \boxed {ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}}\Big[\log p_{\theta}(x|z) \Big] - \mathbb{D}_{KL} \Big({q_{\phi}(z|x)} \ || \ p_{\theta}(z)\Big) } \ (2) $$ 

* The first term 
($$ \mathbb{E_{q_{\phi}(z|x)}} \Big[\log p_{\theta}(x|z) \Big] $$) 
represent the likelihood and the quality of our decoder. Note here, that the expectation is taken with respect to the sample z, which is sampled from $$ q_{\phi} (z|x) $$ 
because we want a meaningful latent vector.

* The second term 
($$ \mathbb{D}_{KL} \Big({q_{\phi}(z|x)} \ || \ p_{\theta}(z)\Big) $$)
the quality of our encoder and act as a regularizer, such that the encoded latent vector will follow our choice of a normal distribution. 

### *Reparameterization Trick*

We found a valid lower bound for our distribution which is our $$ELBO$$ which is a function of $$\theta$$ and $$\phi$$. 

So we can wirte that for a given dataset, the $$ELBO$$ objective can be expressed as  :

$$ \mathcal{L}_{\theta,\phi} \big(x\big) = \sum_{x_i} ELBO(x_i) $$

Now let's look at is gradient  :

* First with respect to $$\theta$$ :
From $$ (1) $$, we can see that since $$q_{\phi}(z|x) $$ does not depend on theta, the gradient of the objective is : 
   $$\nabla_{\theta} \mathcal{L}_{\theta,\phi} \big(x\big) \simeq \nabla_{\theta} \ log \ p_{\theta} (x,z) $$

* With respect to $$\phi$$ :  
This is where it gets tricky, because the $$ ELBO$$ expectation is taken with respect to  $${q_{\phi}(z|x)}$$ which is a function of $$\phi$$, we cannot really compute its gradient. 









