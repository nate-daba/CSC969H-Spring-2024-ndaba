# Paper Summaries for CSC696H: Probabilistic Methods in Machine Learning

This is a collection of summary notes for papers assigned as weekly readings from the course CSC696H: Probabilistic Methods in ML, taught by [Prof. Jason Pacheco](http://www.pachecoj.com/) in Spring 2024. The purpose of these summaries is to critically engage with the material, demonstrating an understanding and critique of the papers.

## Content

1. [Approximate Bayesian Computation (ABC)](#paper-1)
2. [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation](#paper-2)
3. [Weight Uncertainty in Neural Networks](#paper-3)
4. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](#paper-4)
5. [Variational Dropout and the Local Reparameterization Trick](#paper-5)
6. [Deep Variational Information Bottleneck](#paper-6)
7. [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](#paper-7)

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

**TL;DR**: The paper introduces an algorithm for learning a probability distribution on the weights of a neural network (NN), called Bayes by Backprop.

### Strengths
The paper begins by motivating the need for introducing uncertainty on the weights of a NN which is 1) plain feedforward NNs being prone to overfitting, and 2) NNs making overly confident decisions because of being unable to account for the uncertainty in the training data. The paper also formulates and clearly describes a mechanism for obtaining unbaised estimates of gradients of the cost function with respect to the parameters $\theta$. Another strength of the paper is removing the constraint that the complexity cost should have a closed form. By removing this constraint, the proposed method provides flexibilty when choosing variational posterior and prior distributions. Logical justifications for some of the design choices is also provided. E.g. choice of a scale misture of two Gaussian densities as the prior, choice of scheme for weighting the complexity cost relative the likelihood cost, etc. Moreover, this work differentiates itself from prior works in the sense that it tries to learn distributions on weights of NNs compared to most prior works that attempt to learn distributions on stochastic hidden units.

### Improvements
The impact of choosing a non-diagonal Gaussian as the variational posterior on performance can be investigated as it can help capture correlations between weights, which might be present in the true posterior. By using a non-diagonal Gaussian, the model might learn dependencies between weights. This can be particularly important if the weights are not independent in reality, as capturing these dependencies can potentially lead to better generalization and a more robust model.


### Discussions points
While experimenting with the Bandits on Mushroom task, the reported number of times the weights are sampled from the learned posterior i.e. 2, seems a very small number to estimate the expected reward. Perhaps using a larger number of samples can offer better reward estimates and thus might result in faster convergence in terms of cumulative regret.
  
## Paper 4: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://proceedings.mlr.press/v48/gal16.pdf) <a name="paper-4"></a>

**TL;DR**: This paper presents a new way of understanding dropout, a popular neural network regularization technique, as approximate Bayesian inference in deep Gaussian processes.

### Strengths
* Provides an indepth analysis of the relationship between dropout and a deep Gaussian process.
* Supports claims such as improvements in RMSE and uncertainty estimation with results from well crafted extensive experiments.
* Demonstrates the importance of quantifying and interpreting uncertainty. E.g. increasing standard deviation for test points far from training data in the MC dropout model on the Mauna Loa $CO_{2}$ concentrations dataset.
  
### Improvements
Some critical details of the algorithm are pushed to the appendix rendering the paper a bit inaccessible. For example, the approximation of a deep Gaussian process with $L$ layers and covariance function $\textbf{K}(x, y)$ by placing a variational distributions over the components of a spectral decomposition of the GPs' covraince functions was something that deserved a few lines of derivation to clarify the approximation.
  
### Discussions points
* In the experiment for modelling uncertainity in regression tasks, the number of forward passes performed (i.e. 1000) may not scale well with modern neural networks calling for a more efficient approach.
* In table 1, although the dropout results outperform the VI and PBP methods in terms of average RMSE, the average std. errors for the dropout results is higher for almost all datasets.
* It is not mentioned how many forward passes were used to obtain the results in table 1 for the dropout method.

## Paper 5: [Variational Dropout and the Local Reparameterization Trick](https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf) <a name="paper-5"></a>

**TL;DR**: The paper presents a local reparametriziation techniaue for reducing the varaince of stochastic gradients for variational Bayesian inference (SGVB) of a posterior over model parameters.

### Strengths
* The speed of the optimization from the proposed method is on the same level as a fast dropout.
* The method also strikes a balance between flexibility (i.e. because of the flexibly parametrizied posteriors) and optimization speed.
* Provides a simple mechanism for adaptively learning the dropout rate in response to the data.
* Clearly dmonstrates how the local reparametrization trick leads to a more computationally efficient gradient estimator that has a lower variance compared to the regular SGVB estimator.
* The proposed adaptive variational Gaussian dropout is simple and incurs a very negligable computational cost.
  
### Improvements
* The performance of the proposed method was tested only on simple and small datasets (i.e. MNIST and CIFAR-10) and networks (3 layer fully connected network and 2 lyaer CNN network). It would be better to test the effectiveness of the method on more diverese tasks (not just classification) and larger networks that are of practical importance.

### Discussions points
* In the results section, in part (a) of Figure 1, we see that Variational (A) (i.e. when correlated weight noise is introduced) performs way better than Variational (B) (i.e. when independent weight noise is introduced). What is the explanation for this? Does this mean that we should always go with Variational (A) approach?
* What is the reasoning behind the observation that downscaling the KL divergence part of the variational objective yields a better result in terms of test error other than the argument that it just prevents underfitting?

## Paper 6: [Deep Variational Information Bottleneck](https://arxiv.org/pdf/1612.00410.pdf) <a name="paper-6"></a>

**TL;DR**: The paper presents a variational approximation to the information bottleneck (IB) by establishing a lower bound on and approximating the IB objective.

### Strengths
* Gives a good description of the original information bottleneck (IB) problem.
* Points out the challenge of the vanilla IB principle.
* Provides a nice formulation of variational inference to construct a lower bound on the IB objective and how to optimize it using stochastic gradient descent.
* Demonstrates the regularizing effect of VIB by comparing with prior regularization techniques at the time.
* Gives a good analysis of the effects of the hyperparameter $\beta$ and the embedding size $K$ on training and test performance. One interesting intution revealed by the authors is how values of $\beta$ that give best results correspond to the events where the mutula information between the stochastic encoding $Z$ and the images $X$ is between 10 to 100 bits.

### Improvements
* It would be interesting to provide the reasoning for the inverted U-shape accuracy curve in Figure 6. I.e. why do we see a decline in accuracy as $\beta$ increases beyond a certain value e.g. $10^{-5}$ despite Figure 5 showing the opposite effect of increasing $\beta$ on accuracy?
* Can there be a benefit to learning $\beta$ (e.g. performing variational inference on $\beta$) instead of just treating it as a hyperparameter?

### Discussions points
* For experiments on MNIST, it is mentioned that using more than a single Monte Carlo sample of $z$ when predicting $y$ yields better results as evident from Figure 1(a). However, at how much computational cost does this benefit come compared to prior regularization methods?

## Paper 7: [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://proceedings.neurips.cc/paper_files/paper/2016/file/7c9d0b1f96aebd7b5eca8c3edaa19ebb-Paper.pdf) <a name="paper-7"></a>

**TL;DR**: 

### Strengths


### Improvements


### Discussions points

