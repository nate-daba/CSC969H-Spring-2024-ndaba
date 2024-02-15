# Paper Summaries for CSC696H: Probabilistic Methods in Machine Learning

This is a collection of summary notes for papers assigned as weekly readings from the course CSC696H: Probabilistic Methods in ML, taught by [Prof. Jason Pacheco](https://link-to-professor-website.com) in Spring 2024. The purpose of these summaries is to critically engage with the material, demonstrating an understanding and critique of the papers.

## Content

1. [Approximate Bayesian Computation (ABC)](#paper-1)
2. [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation](#paper-2)
3. [Weight Uncertainty in Neural Networks](#paper-3)

## Paper 1: [Approximate Bayesian Computation (ABC)](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1002803&type=printable) <a name="paper-1"></a>

**TL;DR**: Approximate Bayesian Computation (ABC) is a set of techniques for statistical inference that does not require the explicit computation of the likelihood function.

### Strengths

The paper provides a clear explanation of the ABC rejection algorithm. It detailed how, given a prior distribution of model parameter $\theta$, ABC can be used to estimate the posterior distribution of parameter values without explicitly evaluating the likelihood function. This estimated posterior can then, in turn, be used to estimate the most likely parameter $\theta$ given the data **$D$**. Figure 1 in the paper succinctly depicts this parameter estimation process. Moreover, it provides a simple and practical example that demonstrated how ABC could be used to approximate the posterior distribution in a bistable system characterized by a hidden Markov model (HMM) subject to measurement noise. The application of ABC beyond parameter estimation is also showcased, demonstrating how ABC is used to compute posterior probabilities of different models and compare the plausibilities of these models using their posterior ratios. Additionally, some common shortcomings of the ABC method and corresponding works that attempt to address these shortcomings are discussed.

### Improvements

In the topic regarding model comparison with ABC, it is not clear what "The relative acceptance frequencies for the different models can approximate the posterior distribution of theses models" exactly means. Does the "relative acceptance frequency" refer to the acceptance frequencies of parameters from models? Something akin to Figure 1 would have been great to explain this process.

In the results depicted in Figure 3, the number of ABC simulations used for the cases of "ABC with $\epsilon$ = 0 and full data", "ABC with $\epsilon$ = 0 and summary statistic", and "ABC with $\epsilon$ = 2 and summary statistic" is not mentioned. This information is crucial for comparing and contrasting the results from the "worked example" case.

### Discussion points

* What are some other distance measures $\rho$ that prior works have explored? Only Euclidean distance and absolute difference were mentioned as examples.
* How do these distance measures work with different summary statistics **$S$**?
* Is it possible to learn these distance metrics using modern methods such as deep metric learning?

## Paper 2: [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation](https://arxiv.org/pdf/1605.06376.pdf) <a name="paper-2"></a>

**TL;DR**: The paper proposes a new approach to likelihood-free inference based on Bayesian conditional density estimation.

### Strengths

The paper does a good job of:
* Describing prior work on inference methods for simulator-based models. E.g. Rejection ABC, MCMC-ABC, and SMC-ABC.
* Pointing out some shortcomings of prior work. E.g. Noisy computation in representing parameter estimates as a set of samples, impracticality of simulations as ε-tolerance is reduced, etc
* Proposing a parametric approach to likelihood-free inference to address the drawbacks of previous methods.
* Providing a principled approach to learning the posterior $\hat{p}(\theta | x = x_{0})$ 
* Testing the proposed approach on various tasks. E.g. Mixture of two Gaussians, Bayesian linear regresstion, etc.

### Improvements

In the description of the results for the Lotka–Volterra predator-prey population model, it would have been better if estimates of log of the remaining three parameters (i.e. $log\ \theta_{2}$, $log\ \theta_{3}$, and $log\ \theta_{4}$) are also reported instead of just $log\ \theta_{1}$. This can show whether or not the learned posterior offers better estimates with better confidence accross all parameters compared to ABC methods. The same goes for results in Figure 4 of M/G/1 queue model where methods are compared only for estimates of $\theta_{2}$.

### Discussions points

* In proposition 1 on page 2 of the paper, how would the proposition change if the set of pairs ($\theta_{n}$, $x_{n}$) are not independent?
* How does the proposed approach handle higher dimensional data (i.e for the case when dim(x) >> 10) which can be common in many practical settings?

## Paper 3: [Weight Uncertainty in Neural Networks](https://proceedings.mlr.press/v37/blundell15.pdf) <a name="paper-3"></a>

