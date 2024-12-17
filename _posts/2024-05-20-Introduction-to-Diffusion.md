---
usemathjax : true
title : Introduction to diffusion 

figureTemplate: '*$$figureTitle$$ $$i$$*$$titleDelim$$ $$t$$'
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

So, if we want to train a generative model, we will want maximize the likelihood of the observed data. But we also want the latent space to be general enough to be able to give us a strong representations of the observed data.

>In general, with generative modeling, we want to learn a model to maximize the likelihood of 
>$$p(x)$$ 
>of all observed 
>$$x$$
>, but because it is difficult to oompute directly, we consider a latent variable $$z$$ that we will use to recover our likelihood 

Now let's consider some distributions : the one over the observation data, our prior :

* $$p(x)$$

* $$p(z)$$ 
the distribution of the latent variable and for the sake of simplicity let's suppose it is a unit-variance gaussian 
$$p(z)=\mathcal{N}(0,\,1)$$

* $$p(z|x)$$ 
that describes the distribution of the encoded variable given the decoded one 

* $$ p(x|z) $$ 
that describes the distribution of the decoded variable given the encoded one .

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
> The integral becomes : $$ p(x_i)= \displaystyle \int \sum_{j} p(z_{j})\prod_{i=1}^{n} p(x_i|z_j) \, \mathrm{d}z_j $$

To make this more tractable, we will need to do some approximation. 

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

$$
\begin{align} \boxed{\log(p_{\theta}(x)) \ge \mathop{\mathbb{E_{q_{\phi}(z|x)}}}\Big[\log  \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)}\Big]} \\ 
\end{align}
\tag{1}$$ 


Let's denote 
$$\begin{align}
\boxed{ELBO(x) =\mathbb{E_{q_{\phi}(z|x)}} \Big[ \log \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)} \Big]} \\
\end{align}
\tag{2}
$$ 

From this derivation, we were able to directly arrive at our lower bound directly, but it does not really give us the intuition on why this ELBO is a valid lower bound and why optimize it.

Now, we will try to show that our denoted ELBO is a valid lower bound and its relationship with the log likelihood of the data. 

$$
\begin{align}
\log \ p_{\theta}(x) &=\log \ p_{\theta}(x) \displaystyle \int q_{\phi}(z|x) \, \mathrm{d}z \\
&=\displaystyle \int q_{\phi}(z|x)log \ p_{\theta}(x)  \, \mathrm{d}z \\
 &= \mathbb{E_{q_{\phi}(z|x)}} \Big[ \log {p_{\theta}(x)} \Big] \\
 &= \mathbb{E_{q_{\phi}(z|x)}} \Big[ \log  \dfrac{p_{\theta}(x,z)}{p_{\theta}(z|x)} \Big] \\
 &= \mathbb{E_{q_{\phi}(z|x)}} \Big[ \log  \dfrac{p_{\theta}(x,z){q_{\phi}(z|x)}}{p_{\theta}(z|x){q_{\phi}(z|x)}} \Big] \\
 &= ELBO + \mathbb{E_{q_{\phi}(z|x)}} \Big[ \log \dfrac{q_{\phi}(z|x)}{p_{\theta}(z|x)}\Big] \\
\log \ p_{\theta}(x)& = ELBO + \mathbb{D}_{KL} \Big({q_{\phi}(z|x)} \ || \ p_{\theta}(z|x)\Big) 
\end{align}$$


And by definition because $$KL$$ divergence is always positive :

$$\log \ p_{\theta}(x) \geq ELBO $$

So the difference between $$ELBO$$ and our evidence is clearly non negative, thus the $$ELBO$$ can never exceed our evidence and is a valid lower bound.

Now let's look more closely at what is this ELBO.  

By Bayes theorem, we can develop our ELBO to look like : 
$$ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}} \Big[ \log \dfrac{p_{\theta}(x,z)}{q_{\phi}(z|x)} \Big]$$
$$ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}}\Big[ \log p_{\theta}(x|z) \Big]+
 \mathbb{E_{q_{\phi}(z|x)}}\Big[\log\dfrac{p(z)}{q_{\phi}(z|x)}\Big]$$

Finally :
$$\begin{align} 
\boxed {ELBO(x)=\mathbb{E_{q_{\phi}(z|x)}}\Big[\log p_{\theta}(x|z) \Big] - \mathbb{D}_{KL} \Big({q_{\phi}(z|x)} \ || \ p_{\theta}(z)\Big) } \\
\end{align} \tag{1}$$ 

* The first term 
($$ \mathbb{E_{q_{\phi}(z|x)}} \Big[\log p_{\theta}(x|z) \Big] $$) 
represent the quality of our decoder (the reconstruction term). Note here, that the expectation is taken with respect to the sample z, which is sampled from $$ q_{\phi} (z|x) $$ 
because we want a meaningful latent vector.

* The second term 
($$ \mathbb{D}_{KL} \Big({q_{\phi}(z|x)} \ || \ p_{\theta}(z)\Big) $$)
represent the quality of our encoder and act as a regularizer, such that the encoded latent vector will follow our choice of a unit variance gaussian. 

### *Reparameterization Trick*

We found a valid lower bound for our distribution which is our $$ELBO$$ which is a function of $$\theta$$ and $$\phi$$. 

So we can wirte that for a given dataset, the $$ELBO$$ objective can be expressed as :

$$ \mathcal{L}_{\theta,\phi} \big(x\big) = \sum_{i} ELBO(x_i) $$

Now let's look at is gradient  :

* First with respect to $$\theta$$ :
From $$ (1) $$, we can see that since $$q_{\phi}(z|x) $$ does not depend on theta, the gradient of the objective is : 
   $$\nabla_{\theta} \mathcal{L}_{\theta,\phi} \big(x\big) \simeq \nabla_{\theta} \ log \ p_{\theta} (x,z) $$

* With respect to $$\phi$$ :  
This is where it gets tricky, because the $$ ELBO$$ expectation is taken with respect to  $${q_{\phi}(z|x)}$$ which is a function of $$\phi$$, so we cannot really compute its gradient as easily as for $$ {\theta} $$ . 

This is where the reparametrization trick comes in handy. Intead of sampling $$z$$
from 
$$z \sim q_{\phi} (z|x)$$ 
where 
$$q_{\phi} (z|x)= \mathcal{N} (z|\mu(x) , \sigma^2(x)I) $$ 
is a Gaussian with mean
$$\mu$$ 
and covariance matrix 
$$\sigma^2I$$ 
(we asssume equal variance).

We express 
$$z \sim q_{\phi} (z|x)$$ 
as some differentiable tranformation of another random varaible 
$$\epsilon$$
, let 
$$z=g(\epsilon, x, \phi) $$
and draw 
$$\epsilon \sim p(\epsilon) $$
.In another term 
$$z=\mu(x) + \sigma^{\frac{1}{2}}(x)\epsilon$$ 
where 
$$\epsilon \sim \mathcal{N}(0,I)$$

The idea is shown below in Figure 2.1.

{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/vae_repara.png)
{: refdef}

{:refdef: style="text-align: center;"}
<figcaption> Figure 2.1: Illustration of reparametrization trick (Source : Diederik P. Kingma and Max Welling (2019), “An Introduction to
Variational Autoencoders”, Foundations and Trends R© in Machine Learning)</figcaption>
{: refdef}

### *Training*  
After everything said, the loss function in our case is simply (with Monte-Carlo simulation, with D the number of samples in our dataset):

$$\mathcal{L}_{\theta,\phi} \simeq \frac{1}{D} \sum_{i=1}^{D} log \ p_{\theta}(x^{i}| z^{i}) - \mathbb{D}_{KL}\big(q_{\phi}(z|x^{i})\ ||p_{\theta}(z)\big)   $$

So, we are searching for the $$\theta$$ and $$\phi$$ which will maximize $$  \mathcal{L}_{\theta,\phi} $$ 
. In another word, we want to find  :

$$ \boxed {\theta^{*}, \phi^{*} = \arg \max_{\theta, \phi} \mathcal{L}_{\theta, \phi}}  \tag{3}$$


>So to resume what we saw : 
>- AE are very at obtaining a compressed representation of the input data. But its features space is not continous, which cannot provide a rich semantic correlation of the input space 
>
> - VAE are better than AE, in this point with latent space regularization. However, VAE suffers from the case of posterior collapse, where the posterior of the latent variable is equal to the prior. 

## Diffusion Models 

In this section, I will try to discuss diffusion in the sense of Variational diffusion Models or DDPM (denoising diffusion probabilistic model) [[2]](#references). 

Variational diffusion models can be simply seen as a markovian hierachical variational encoders (a VAE with multiple latent variables instead of just one, and where each latent is conditioned on all previous latents), with some conditions :

* The latent dimension is the same as the data 
* The structure of our encoder is a pred-defined Gaussian dependent on the output of the previous step


As resumed by the figure below where:

* $$x_0$$ 
is the original image
* $$x_T$$ 
is the latent variable, we want $$x_T\sim \mathcal{N}(0,I)$$
* $$x_1,...,x_{T-1}$$ 
are intermediate states, also latent variable but we don't want them to be white noise


{:refdef: style="text-align: center;"}
![_config.yml]({{ site.baseurl }}/images/Diffusion_intro_1.png)
{: refdef}
{:refdef: style="text-align: center;"}
<figcaption> Figure 3 : Principle of Diffusion</figcaption>
{: refdef}

With the same reasoning that we did with VAE, because the transition and reverse distribution is not known, we will try to approximate them. 



So the reverse transition
$$ p(x_t|x_{t+1}) $$
is approximated by a Gaussian 
$$ p_{\theta}(x_t|x_{t+1}) $$
and the transition
$$ p(x_t|x_{t-1}) $$ 
by 
$$ q_{\phi}(x_t|x_{t-1}) $$


For the transition distribution, let's define it as follows :
$$ q(x_t|x_{t-1})=\mathcal{N}(x_t\ |\ \sqrt \alpha_t x_{t-1}, (1-\alpha_t)I) $$

The coefficients are chosen such that the variance of the latent variables stays at a similar scale. Note here, that our $$ q_{\phi} $$ is fully known, as it is modeled as a Gaussian with definitive mean and variance for each timestep. So in diffusion(DDPM), we are primarly interested in the reverse conditional distribution $$p_{\theta}$$ which we will use to simulate new data



> But why this choice of 
>$$\sqrt\alpha_t $$ 
> and $$(1-\alpha_t)$$
> you might ask yourself ? 
>
>We know, that since our transition gaussian is in the form of 
> $$q(x_t|x_{t-1})=\mathcal{N} (x_t|ax_{t-1}, b^2I) $$
> with $$a,b \in \mathbb{R}$$
>
>With the reparametrization trick, we can define $$x_t$$ as :
>
>$$x_t=ax_{t-1} + b\epsilon_{t-1}$$
>where $$\epsilon_{t-1} \sim \mathcal{N}(0,I)$$
>
>Now we can easily show that :
>
> $$x_t=a^tx_0 + b\sum_{k=0}^{t-1}a^{(t-1)-k}\epsilon_k$$
>
> The second term is just a sum of independant gaussian let's denote it as  $$z_t$$
>
> Because of the fact that each 
> $$ \epsilon_k   $$ 
>has zero mean  
> $$\mathbb{E}[z_t]=0$$
>
> For the variance, since 
> $$Var[z_t]=Cov(z_t,z_t)$$
>,and because of the bilinearity property of the covariance : 
>
>$$\begin{align}
>Cov(z_t,z_t)&=Cov(b\sum_{k=0}^{t-1}a^{(t-1)-k}\epsilon_k,b\sum_{l=0}^{t-1}a^{(t-1)-k}\epsilon_l) \\
>&=b^2\sum_{i=0}^{t-1}\sum_{j=0}^{t-1}Cov(a^{(t-1)-k}\epsilon_k,a^{(t-1)-l}\epsilon_l) \\
>\end{align}$$
> Because each term is independant and the 
>$$\epsilon_k \sim \mathcal{N}(0,I)$$
>
>$$\begin{align}
>Cov(z_t,z_t)&= b^2\sum_{k=0}^{t-1}Var(a^{(t-1)-k}\epsilon_k )\\
>&= b^2 \sum_{k=0}^{t-1}a^{2((t-1)-k)}\\
>&= b^2 \frac{1-a^{2t}}{1-a^2}\end{align}$$
>
>
> So for any 
> $$ 0<a<1 $$
> and when 
> $$ t \to {+\infty} $$
> $$ \lim_{t \to {+\infty}} Var(z_t) = \frac{b^2}{1-a^2} $$
>
> Because we want our distribution of 
> $$x_t$$
> to approach 
> an isotropic gaussian 
> as t grows
>,
> $$  \lim_{t \to {+\infty}} Var(z_t) =I$$ 
>
> Then 
> $$b=\sqrt{1-a^2}$$
>,
> so we can let, 
> $$a=\sqrt\alpha$$
>, for any 
> $$\alpha \in \left[ 0,1\right]$$
>then 
>$$ b= \sqrt{1-\alpha}$$
>.
>
>Finally we can write  : 
> $$x_t=\sqrt \alpha_t x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}$$


### *ELBO*

Before deriving the ELBO for our case, to make the process more tractable let's impose a Markovian structure (memoryless) where: 

$$ q_{\phi} (x_{0:T}|x_0)= q(x_0) \prod_{t=1}^T q_{\phi}(x_t|x_{t-1}) $$

$$ p_{\theta} (x_{0:T}) = p_{\theta}(x_T) \prod_{t=1}^{T} p_{\theta} (x_{t-1}|x_t) $$

For the ELBO, we will follow the same steps as we did for the VAE, with very few differences.
$$ $$

$$\begin{align}
\log p_{\theta}(x_{0:T})&=\int \frac {q_{\phi}(x_{1:t}|x_0)\ p_{\theta}(x_{0:t}) }{q_{\phi}(x_{1:t}|x_0)} \ \ \mathrm{d}{x_{1:T}} \\
&= \log \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big(\ \frac{p_{\theta}(x_{0:t}) }{q_{\phi}(x_{1:t}|x_0)} \ \Big) \\
&\geq  \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_{0:t}) }{q_{\phi}(x_{1:t}|x_0)} \ \Big] \\
\end{align}$$

With the things defined before we can further simplify the above expression to obtain that : 

$$\begin{align}
\log p_{\theta}(x_{0:t})
&\geq  \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_{0:t}) }{q_{\phi}(x_{1:t}|x_0)} \ \Big] \\
\\
&= \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) \prod_{t=1}^{T} p_{\theta} (x_{t-1}|x_t) }{\prod_{t=1}^T q_{\phi}(x_t|x_{t-1})}\ \Big] \\
\\
&= \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) p_{\theta}(x_0|x_1) \prod_{t=2}^{T} p_{\theta} (x_{t-1}|x_t) }{q_{\phi}(x_T|x_{T-1})\prod_{t=1}^{T-1} q_{\phi}(x_t|x_{t-1})}\ \Big] \\
\\
&= \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) p_{\theta}(x_0|x_1) \prod_{t=1}^{T-1} p_{\theta} (x_{t}|x_{t+1}) }{q_{\phi}(x_T|x_{T-1})\prod_{t=1}^{T-1} q_{\phi}(x_t|x_{t-1})}\ \Big] \\
\\
&=  \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[ \log \ p_{\theta}(x_0|x_1)\Big] + \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[ \log \ \frac{p_{\theta}(x_T)}{q_{\phi}(x_T|x_{T-1})} \Big] +
\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[ \log \ \frac{ \prod_{t=1}^{T-1} p_{\theta} (x_{t}|x_{t+1}) }{\prod_{t=1}^{T-1} q_{\phi}(x_t|x_{t-1})}\ \Big] \\
\\
&=  \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[ \log \ p_{\theta}(x_0|x_1)\Big] + \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[ \log \ \frac{p_{\theta}(x_T)}{q_{\phi}(x_T|x_{T-1})} \Big] +
\sum_{t=1}^{t-1} \mathbb{E_{q_{\phi}(x_{t}  |   x_0)} } \Big[ \log \ \frac{ p_{\theta} (x_{t}|x_{t+1}) }{ q_{\phi}(x_t|x_{t-1})}\ \Big] \\
\end{align}$$

The conditioning in the first term can be simplified to 
$$ x_1|x_0 $$
.By applying the same reasoning to the second and last terms, we obtain:

- $$ \mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[ \log \ p_{\theta}(x_0|x_1)\Big] + \mathbb{E_{q_{\phi}(x_{T}, x_{T-1}  |   x_0)} } \Big[ \log \ \frac{p_{\theta}(x_T)}{q_{\phi}(x_T|x_{T-1})} \Big] +
\sum_{t=1}^{t-1} \mathbb{E_{q_{\phi}(x_{t},x_{t+1},x_{t-1}  |   x_0)} } \Big[ \log \ \frac{ p_{\theta} (x_{t}|x_{t+1}) }{ q_{\phi}(x_t|x_{t-1})}\ \Big] $$ 

$$\\$$

Finally we obtain for our ELBO : 

$$
\begin{align}

\boxed{ELBO_{\phi,\theta}(x) = \underbrace{\mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[ \log \ p_{\theta}(x_0|x_1)\Big]}_\textbf{Reconstruction term} -
\underbrace{\mathbb{E_{q_{\phi}(x_{T-1} |   x_0)} }\Big[ \ {\mathbb{D_{KL}} \Big[  {q_{\phi}(x_T|x_{T-1})} \ || \ p_{\theta}(x_T) \ \Big]}\Big]}_\textbf{Prior Matching} +
\underbrace{\sum_{t=1}^{T-1}\mathbb{E_{q_{\phi}(x_{t-1},x_{t+1}  |   x_0)} } \Big[ \ \mathbb{D_{KL}} \Big[ \   q_{\phi}(x_t|x_{t-1}) || p_{\theta} (x_{t}|x_{t+1}) \ \Big] \ \Big]} _\textbf{Consitency Term}}

\end{align}
$$

* The Reconstruction term can be interpreted the same way as in the vanilla VAE. We measure how good our neural network can recover 
$$x_0$$ 
from 
$$x_1$$
sampled from 
$$ q_{\phi} (x_1|x_0) $$ 
.

* The second term, the prior matching describe the final block. This term brings closer our
$$ q_{\phi} (x_{T}|x_{T-1}) $$
to an isotropic gaussian.

* The third term, the consistency term describes the intermediates states, it measures the deviation to make 
$$ x_t $$ 
consistant from both the transiton and reverse state. 


Observe here that for the consistency term, we sample from a joint distribution 
$$q_{\phi}(x_{t-1},x_{t+1}  |   x_0)$$
meaning we use the random variables 
$${x_{t-1},x_{t+1}} $$
to draw the current sample 
$$x_t$$ 
which may seem a little odd, and also because the two distributions are moving in opposite direction. 

The ELBO term above that we found, can be computed without any problem. However, it might be suboptimal, because the consitency term is computed as an expectation of two random variables  : $$ {x_{t-1},x_t} $$
and also because we sum over $$ T-1 $$ term, over large T the ELBO might have a high variance. 


Let us try to resolve all of this : 

By Baye's Rule we know we can express our opposing flow : 
$$ 
\begin{equation}
q_{\phi}(x_{t}|x_{t-1},x_0)= \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)}
\end{equation}
$$


Now let us rederive to obtain our new ELBO.


$$
\begin{align}

log p_{\theta}(x_{0:t}) &\geq  \mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_{0:t}) }{q_{\phi}(x_{1:t}|x_0)} \ \Big] \\
&=\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) \prod_{t=1}^{T} p_{\theta} (x_{t-1}|x_t) }{\prod_{t=1}^T q_{\phi}(x_t|x_{t-1})}\ \Big] \\
&=\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T)  p_{\theta}(x_0|x_1) \prod_{t=2}^{T} p_{\theta} (x_{t-1}|x_t) }{ q_{\phi}(x_1|x_{0}) \prod_{t=2}^T q_{\phi}(x_t|x_{t-1})}\ \Big] \\

&=\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) p_{\theta}(x_0|x_1) \prod_{t=2}^{T} p_{\theta} (x_{t-1}|x_t) }{ q_{\phi}(x_1|x_{0}) \prod_{t=2}^T \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)} } \ \Big] \\

&=\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) p_{\theta}(x_0|x_1)}{ q_{\phi}(x_1|x_{0})}  + \prod_{t=2}^{T}  \log \ \frac{p_{\theta} (x_{t-1}|x_t) }{ \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)} } \ \Big] 


\end{align}
$$

Let's consider the second term : 

$$
\begin{align}

\prod_{t=2}^{T}  \Big[\log \ \frac{p_{\theta} (x_{t-1}|x_t) }{ \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)} } \ \Big] 
&=\prod_{t=2}^{T}  \Big[\log \ \frac{p_{\theta} (x_{t-1}|x_t) }{ \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)} } \ \Big] \\
&=\prod_{t=2}^{T}  \Big[\log \ \frac{p_{\theta} (x_{t-1}|x_t) }{q_{\phi}(x_{t-1}|x_t,x_0) }  \cdot \prod_{t=2}^{T}  \frac {q_{\phi}(x_{t-1}|x_0)}{ q_{\phi}(x_t|x_0)} \Big] \\
&=\prod_{t=2}^{T}  \Big[\log \ \frac{p_{\theta} (x_{t-1}|x_t) }{q_{\phi}(x_{t-1}|x_t,x_0) }  \cdot \frac {q_{\phi}(x_{1}|x_0)}{ q_{\phi}(x_T|x_0)} \Big ]

\end{align}
$$


Then going back to equation $$(32)$$, 
by cancelling $$q_{\phi}(x_1|x_0)$$
 we can see that : 


$$
\begin{aligned}

\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} } \Big[\log \ \frac{p_{\theta}(x_T) p_{\theta}(x_0|x_1)}{ q_{\phi}(x_1|x_{0})} \Big] + \prod_{t=2}^{T}  \Big[\log \ \frac{p_{\theta} (x_{t-1}|x_t) }{ \frac{q_{\phi}(x_{t-1}|x_t,x_0) q_{\phi}(x_t|x_0)}{q_{\phi}(x_{t-1}|x_0)} } \ \Big] 

&=\mathbb{E_{q_{\phi}(x_{1:T}  |   x_0)} }  \Big[ \ \log \frac{p_{\theta}(x_T)}{ q_{\phi}(x_T|x_{0})} +   \log \ p_{\theta}(x_0|x_1) -
 
\sum_{t=2}^{T} \log  \frac{q_{\phi}(x_{t-1}|x_t,x_0)}{ p_{\theta} (x_{t-1}|x_{t})} 

\Big] \\

&=\mathbb{E_{q_{\phi}(x_{T}  |   x_0)} }  \Big[ \log \ \frac{p_{\theta}(x_T)}{ q_{\phi}(x_T|x_{0})} \Big] + \mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[\log \ p_{\theta}(x_0|x_1) \Big] - \sum_{t=2}^{T} \mathbb{E_{q_{\phi}(x_{t},x_{t-1}  |   x_0)} } \Big[\log \frac{q_{\phi}(x_{t-1}|x_t, x_0)}{ p_{\theta} (x_{t-1}|x_{t})}      \Big] \\

&= \underbrace{\mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[\log \ p_{\theta}(x_0|x_1) \Big]}_\textbf{Reconstruction term} - \underbrace{\mathbb{D_{KL}} \Big[ \   q_{\phi}(x_T|x_{0}) || p_{\theta} (x_{T}) \Big]}_\textbf{Prior Matching} -

 \underbrace {\sum_{t=2}^{T}\mathbb{E_{q_{\phi}(x_{t}  |   x_0)} } \mathbb{D_{KL}} \Big[ \   q_{\phi}(x_{t-1}|x_{t},x_0) || p_{\theta} (x_{t-1}|x_t) \Big]}_\textbf{Consistency term}

\end{aligned} 
$$

$$ \\[5pt]$$

* For the **Reconstruction term**, it can be interpretated in the same manner as before.
* The **Prior matching term**, can be seen as estimate of how close the final noisified distribution is to our gaussian prior.
* The new **Consistency term**, contrary to before where we asked a forward transition to match with the reverse transition. Now, we use our $$q_{\phi}$$ to construct a reverse transition and match it with
$$p_{\theta}$$

##  Reverse process 

Now that we've derived the new Evidence Lower Bound (ELBO) for our variational diffusion model, let's focus on its essential component: $$ q_\phi(x_{t-1}|x_t, x_0) $$ 
. Our key objective is to demonstrate:

 * This distribution is Gaussian
 * We can compute its mean and variance in closed form


Since we already know that $$q_{\phi} (x_{t-1}|x_t , x_0)  = \mathcal{N}(x_t|\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I}) $$ and via our reparametrization trick drawing samples $$ x_t \sim q_{\phi}(x_t|x_{t-1}) $$ 
can be rewritten as  :

$$
\begin{align}
 x_t&=\sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_{t-1} \\
   &=\sqrt{\alpha_t \cdot \alpha_{t-1}} \cdot x_{t-2} + \sqrt{1-\alpha_t} \cdot \epsilon_{t-1}
      + \sqrt{\alpha_t \cdot (1-\alpha_{t-1})} \cdot \epsilon_{t-2} \\
   &=\sqrt{\alpha_t \cdot \alpha_{t-1}} \cdot x_{t-2} + \sqrt{1-\alpha_t \cdot \alpha_{t-1} } \epsilon_{t-2} \\
   &=.... \\
   &= \sqrt{\prod_{i=1}^{T} \alpha_i} x_0 +\sqrt{1 - \prod_{i=1}^{T}\alpha_i} \epsilon_0 \\
   &=\sqrt{\overline {\alpha}_t} x_0 +\sqrt{ 1 - \overline {\alpha}_t} \epsilon_0\
 \end{align}
$$



The transition between line 2 and line 3 of the equation is achieved through the fact that the sum of two independant Gaussian, remains a Gaussian, with mean being the sum of the two means and the variance being the sum of the two variances. 


To obtain the form of $$q_{\phi}(x_{t-1}|x_t)$$
, using Bayes Theorem. With what we obatined earlier 
($$x_t$$) 
in closed form, It can be determined that :

$$ 
\begin{align}
q_{\phi}(x_{t-1}|x_{t},x_0)&= \frac{q_{\phi}(x_{t}|x_{t-1},x_0) q_{\phi}(x_{t-1}|x_0)}{q_{\phi}(x_{t}|x_0)} \\
&=\frac{\mathcal{N}(x_{t}|\sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I}) \mathcal{N}(x_{t-1}|\sqrt{\overline{ \alpha}_{t-1}} x_{0}, (1-\overline{\alpha}_{t-1})\mathbf{I})} {\mathcal{N}(x_t|\sqrt{ \overline{\alpha}_t} x_{0}, (1-\overline{\alpha}_t)\mathbf{I})} \\

&\propto \exp\left[
    \frac{(x_t - \sqrt{\alpha_t}x_{t-1})^2}{2(1 - \alpha_t)} 
    + \frac{(x_{t-1} - \sqrt{\overline {\alpha}_{t-1}}x_0)^2}{2(1 - \overline{\alpha}_{t-1})} 
    - \frac{(x_t - \sqrt{\overline{\alpha}_t}x_0)^2}{2(1 - \overline{\alpha}_t)}
\right]

\end{align}
$$

From this point, there are two main approaches to determine the mean and variance:  

* Continue expanding the expression until it simplifies into a resulting Gaussian form.  
* Identify the expression inside the exponential function as a quadratic expression. The mean and variance of the resulting Gaussian can then be determined as the minimum of this quadratic function and its derivative, respectively.

After all, we obtain that 
$$q_{\phi}(x_{t-1}|x_t)$$ 
takes the form : 
$$ 
\mathcal{N}(x_{t-1} | \mu_q(x_t,x_0),\Sigma_q(t) \mathbf{I}) 
\\
$$ 

$$\\$$
where

$$
\\
\mu_q(x_t, x_0) = \frac{(1 - \overline{\alpha}_{t-1})\sqrt{\overline{\alpha}_t}}{1 - \overline{\alpha}_t}x_t 
+ \frac{(1 - \alpha_t)\sqrt{\overline{\alpha}_{t-1}}}{1 - \overline{\alpha}_t}x_0
$$

$$
\Sigma_q(t) = \frac{(1 - \alpha_t)(1 - \sqrt{\overline{\alpha}_{t-1}})}{1 - \overline{\alpha}_t} I 


$$


A pertinent observation would be to see that, we obtained an expression of
$$q_{\phi}(x_{t-1}|x_t) $$ 
, which only depends on 
$$ x_t $$ 
and 
$$ x_0 $$
. This implies that there is no "learning" involved for 
$$ q $$
, and no neural network is required to be trained.

Remembering our consistency term from our ELBO : 
$$\underbrace {\sum_{t=2}^{T} \mathbb{D_{KL}} \Big[ \   q_{\phi}(x_{t-1}|x_{t},x_0) || p_{\theta} (x_{t-1}|x_t) \Big]}_\textbf{Consistency term}$$

In order to match $$p_{\theta}(x_{t-1}|x_t)$$ to our gaussian distribution 
$$q_{\phi}(x_{t-1} | x_t) $$
we can model it as a Gaussian with matching mean and variance : 

$$ 
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left( x_{t-1} \mid \mu_\theta(x_t), \sigma_q^2(t) I \right)
$$ 

The mean can be determined via a neural network, and for the variance we choose it to be identical to our true posterior : 
$$q_{\phi}(x_{t-1} | x_t) $$
We only need to learn the mean 
$$\mu_\theta$$ 
our neural network 


Therefore, the KL divergence is simplified to : 
$$
\begin{align}
\mathbb{D_{KL}} \Big[ \   q_{\phi}(x_{t-1}|x_{t},x_0) || p_{\theta} (x_{t-1}|x_t) \Big]
& = \mathbb{D_{KL}}\left( \
\mathcal{N}(x_{t-1} | \mu_q(x_t,x_0),\Sigma_q(t) \mathbf{I} ) \ || \ \mathcal{N}\left( x_{t-1} \mid \mu_\theta(x_t), \sigma_q^2(t) I \right) \
\right) \\
&= \frac{1}{2 \sigma_q^2} \left[ \| \mu_\theta - \mu_q \|^2 \right]
\end{align}
$$


## Training and Inference 

Given our simplified KL divergence, we want to optimize 
$$ \mu_\theta$$ 
to match 
$$ \mu_q $$
In other words, we choose 
$$ \mu_\theta $$ 
to be in form 

$$
\mu_\theta(x_t) = \frac{(1 - \overline{\alpha}_{t-1})\sqrt{\overline{\alpha}_t}}{1 - \overline{\alpha}_t}x_t 
+ \frac{(1 - \alpha_t)\sqrt{\overline{\alpha}_{t-1}}}{1 - \overline{\alpha}_t} \hat{x}_\theta(x_t)
$$


Then our simplified KL divergence, further simplify to  :
$$
\\
\begin{align}
\frac{1}{2 \sigma_q^2} \left[ \| \mu_\theta - \mu_q \|^2 \right]
&=\frac{1}{2 \sigma_q^2} \left[   
\left | \left| \frac{(1 - \alpha_t)\sqrt{\overline{\alpha}_{t-1}}}{1 - \overline{\alpha}_t} *(\hat{x}_\theta(x_t) - x_0 )
\right| \right|^2
\right] \\
&=\frac{1}{2 \sigma_q^2} \frac{(1 - \alpha_t)^2\overline{\alpha}_{t-1}}{(1 - \overline{\alpha}_t)^2} \left[ \| \hat{x}_\theta(x_t) - x_0 \|^2 \right]
\end{align}
$$

Therefore, optimizing a DDPM comes down to training a neural network to predict the original ground truth image, from a empirically choosen noisen version of it. 

Ignoring the constant, remember our ELBO, then it simplify to  :

$$
\\
\begin{align} 
ELBO_{\theta}(x)&=\mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[\log \ p_{\theta}(x_0|x_1) \Big] - \sum_{t=2}^{T}\mathbb{E_{q_{\phi}(x_{t},x_{t-1}  |   x_0)} } \left[ \\ \mathbb{D_{KL}} \Big[ \   q_{\phi}(x_{t-1}|x_{t},x_0) || p_{\theta} (x_{t-1}|x_t) \Big] \right]   \\

&=\mathbb{E_{q_{\phi}(x_{1}  |   x_0)} } \Big[\log \ p_{\theta}(x_0|x_1) \Big] -  \sum_{t=2}^{T}\mathbb{E_{q_{\phi}(x_{t}|   x_0)} } \left[                                     
\frac{1}{2 \sigma_q^2} \frac{(1 - \alpha_t)^2\overline{\alpha}_{t-1}}{(1 - \overline{\alpha}_t)^2} \left[ \| \hat{x}_\theta(x_t) - x_0 \|^2 \right]

\right]

\end{align}
$$


We dropped the term in 
$$ \mathbb{D_{KL}} \Big[ \   q_{\phi}(x_T|x_{0}) || p_{\theta} (x_{T}) \Big] $$
because it remains as a constant, as there is nothing to train. 

Only the reconstruction term  :
$$
\\
\begin{align}
\log p(x_0|x_1) &=  \mathcal{N}(x_0|\mu_\theta(x_1), \sigma_q^2 \mathbb{I}) \propto -\frac{1}{2*\sigma_q^2} \left|
\mu_\theta - \mu_q \right|^2 \\
&= -\frac{1}{2\sigma_q^2} * \frac{(1 - \alpha_1)^2}{(1 - \alpha_1)^2} ||\hat{x}_\theta - x_0 ||^2  \; \text{Because} \; \alpha_0=1, \overline{\alpha_1}=\alpha_1 \\
&=-\frac{1}{2\sigma_q^2}||\hat{x}_\theta - x_0 ||^2  
\end{align}
$$

Then our ELBO becomes : 

$$
ELBO_{\theta}(x)= - \sum_{t=1}^{T}\mathbb{E_{q_{\phi}(x_{t} |   x_0)} }  \left[                                     
\frac{1}{2 \sigma_q^2} \frac{(1 - \alpha_t)^2\overline{\alpha}_{t-1}}{(1 - \overline{\alpha}_t)^2} \left[ \| \hat{x}_\theta(x_t) - x_0 \|^2 \right]

\right]

$$


Considering the above and using Monte-Carlo to approximate the expectation, we can write the optimization problem as  :

$$
\arg\max_\theta \sum_{x_0 \in \mathcal{X}} ELBO(x_t) \\
= \arg\min_\theta \sum_{x_0 \in \mathcal{X}} \sum_{t=1}^{T}\mathbb{E_{q_{\phi}(x_{t} |   x_0)} }  \left[                                     
\frac{1}{2 \sigma_q^2} \frac{(1 - \alpha_t)^2\overline{\alpha}_{t-1}}{(1 - \overline{\alpha}_t)^2} \left[ \| \hat{x}_\theta(x_t) - x_0 \|^2 \right]

\right] \\

=\arg\min_\theta \sum_{x_0 \in \mathcal{X}} \sum_{t=1}^{T} \frac{1}{M} \sum_{i=1}^{M}  \left[                                     
\frac{1}{2 \sigma_q^2} \frac{(1 - \alpha_t)^2\overline{\alpha}_{t-1}}{(1 - \overline{\alpha}_t)^2} \left[ \| \hat{x}_\theta(x_t^{i}) - x_0 \|^2 \right]

\right] \\

$$

where 

$$
x_t^{m} \sim \mathcal{N}(x_t|\sqrt{\overline{\alpha}_t}x_0,(1 - \overline{\alpha}_t)\mathbb{I})
$$

## Equivalent interpretation

Previously, we proved that to train a diffusion model, we can simply train a neural network to predict the original image 
$$
x_0
$$

However, we can reparemetrize 
$$x_0$$
if we re-arrange the equation below :

$$
x_t = \sqrt{\overline{\alpha}_t} x_0 + \sqrt{1 - \overline{\alpha}_t} \epsilon_0
\quad \Rightarrow \quad x_0 = \frac{x_t - \sqrt{1 - \overline{\alpha}_t} \epsilon_0}{\sqrt{\overline{\alpha}_t}}
$$

The injecting it into the true denoising mean 
$$ \mu_q(x_t,x_0) $$ 






## References 

[1] 
Chan, S. H. (2024). Tutorial on Diffusion Models for Imaging and Vision. arXiv preprint arXiv:2403.18103.


[2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851

[3]
Diederik P. Kingma and Max Welling (2019), “An Introduction to
Variational Autoencoders”, Foundations and Trends R© in Machine Learning

[4]
Liu, X., Zhang, F., Hou, Z., Mian, L., Wang, Z., Zhang, J., & Tang, J. (2021). Self-supervised learning: Generative or contrastive. IEEE transactions on knowledge and data engineering, 35(1), 857-876.

[5] Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.

[6] This awesome blog by Lilian Weng : https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
