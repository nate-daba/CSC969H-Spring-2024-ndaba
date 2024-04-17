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
8. [Information Dropout: Learning Optimal Representations Through Noisy Computation](#paper-8)
9. [Auto-Encoding Variational Bayes](#paper-9)
10. [Denoising Diffusion Probabilistic Models](#paper-10)
11. [Denoising Diffusion Implicit Models](#paper-11)
12. [Score-Based Generative Modeling Through Stochastic Differential Equations](#paper-12)
13. [Implicit Generation and Modeling with Energy-Based Models](#paper-13)
14. [How to Train Your Energy-Based Models](#paper-14)
15. [Variational Bayesian Optimal Experimental Design](#paper-15)
16. [On Variational Bounds of Mutual Information](#paper-16)

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

**TL;DR**: InfoGAN innovatively learns disentangled representations in an unsupervised manner by maximizing mutual information between a subset of latent variables and observations, demonstrating meaningful factor discovery across diverse datasets.

### Strengths
* Introduces an information-theoretic extension to GANs that enables the learning of disentangled representations without supervision, addressing a significant challenge in representation learning.
* Demonstrates the ability to discover and separate meaningful factors of variation in data, such as digit styles, facial expressions, and object orientations, across various datasets including MNIST, CelebA, and SVHN.
* Employs a simple yet effective modification to the GAN objective, which adds negligible computational cost, making it a practical approach for enhancing GANs to learn interpretable representations.
* Achieves representation quality competitive with supervised methods, showcasing the potential of unsupervised learning for complex tasks.

### Improvements
* An in-depth analysis on the performance of InfoGAN across datasets with highly entangled or subtle variations could provide insights into its limitations and guide future improvements.
* Investigating the effects of different latent code configurations (e.g., continuous vs. discrete, various dimensionality) on the disentanglement and interpretability of learned representations could optimize the model's effectiveness.
* Incorporating recent developments in GAN architectures and training methodologies could enhance the stability and quality of the representations learned by InfoGAN.

### Discussions points
* Expanding the methodology for evaluating disentanglement and interpretability, possibly by including quantitative metrics and comparisons with human judgments, could offer a more comprehensive assessment of InfoGAN's performance.
* Exploring the application of InfoGAN to other types of data, such as text or audio, could reveal its versatility and potential for unsupervised learning in various domains.
* Assessing the utility of disentangled representations learned by InfoGAN for downstream applications, including classification, regression, or reinforcement learning, could highlight the model's practical implications.

## Paper 8: [Information Dropout: Learning Optimal Representations Through Noisy Computation](http://www.vision.jhu.edu/teaching/learning/deeplearning18/assets/Achille_Soatto-18.pdf) <a name="paper-8"></a>

**TL;DR**: a novel approach that dynamically adjusts noise in neural network activations to improve generalization and achieve disentangled representations, connecting dropout techniques with the Information Bottleneck principle.

### Strengths
* The paper provides a robust theoretical foundation that links Information Dropout with the Information Bottleneck principle, offering insights into dropout techniques and their role in learning disentangled representations.
* Through experiments on datasets like Cluttered MNIST and Occluded CIFAR, the paper demonstrates how Information Dropout can lead to significant improvements in generalization and robustness against nuisances.
* It bridges critical areas in deep learning, including representation learning, information theory, and variational inference, presenting a cohesive understanding of how dropout can aid in learning optimal representations.

### Improvements
* The paper could delve deeper into the computational efficiency and scalability of Information Dropout, particularly in terms of its impact on training dynamics and resource requirements.
* Further research comparing Information Dropout with a wider range of regularization techniques across diverse tasks and models would help validate its effectiveness and adaptability.
* Exploring how Information Dropout performs within different neural network architectures, including recurrent and transformer models, could provide valuable insights into its versatility.

### Discussion Points
* The method's ability to achieve disentangled representations is promising, yet a more in-depth discussion on standardized evaluation metrics for disentanglement would be beneficial.
* Information Dropout introduces a fascinating debate on the optimal balance between noise and signal in neural networks, offering a perspective on how noisy computation might mimic cognitive processes.
* The paper's exploration of data-driven adaptability through Information Dropout invites further discussion on the parallels between neural network information processing and human cognition, potentially uncovering fundamental principles of learning and representation.

## Paper 9: [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) <a name="paper-9"></a>

**TL;DR**: The paper introduces the Stochastic Gradient Variational Bayes (SGVB) estimator and the Auto-Encoding Variational Bayes (AEVB) algorithm, which offer a new method for efficient approximate inference in complex models with intractable posterior distributions.

### Strengths
* Integrates recognition and generative models to optimize the variational lower bound with the SGVB estimator.
* Employs the AEVB algorithm for learning parameters and inference in latent variable models using stochastic gradients.
* Outperforms traditional methods like the wake-sleep algorithm, offering faster convergence and better optimization.

### Improvements
* Expansion to hierarchical generative models and time-series data could be further explored.
* A deeper analysis on the scalability of the AEVB algorithm for complex and large-scale models is warranted.
* Comparison with other variational inference techniques could be broadened and detailed.

### Discussion Points
* It seems that the need for the reparametrization trick is to also get a better gradient estimate with lower variance (i.e. SGVB) than the vanilla estimator in addition to allowing backprop through a "random" variable.
* In the problem statement section, it is not very clear how efficient approximate marginal inference of the variable $x$ can allow us to perform inference tasks such as image denoising, and inpainting since there was no experiemnt or example showing how these tasks actually might make use of the approximate marginal.
* Discussing the limitations and optimization potential of SGVB and AEVB with respect to large datasets.
* Exploring the practical applications of AEVB in recognition tasks and its impact on fields like denoising and visualization.
* Investigating the influence of different neural network architectures within the AEVB framework on the performance of recognition models.

## Paper 10: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) <a name="paper-10"></a>

**TL;DR**: This paper introduces high-quality image synthesis with diffusion probabilistic models, leveraging a novel connection between diffusion probabilistic models, denoising score matching, and Langevin dynamics .

### Strengths
* Demonstrates state-of-the-art image synthesis quality on various benchmarks including CIFAR10 and LSUN datasets .
* Establishes theoretical links between diffusion models, variational inference, and denoising score matching, enhancing understanding of generative models .
* Presents a new perspective on lossy compression through the lens of diffusion models and variational bounds .

### Improvements
* A more comprehensive study on the computational demands and scalability of the diffusion models would be beneficial .
* Would be greate if some failure cases and the plausable reasoning behind them was provided.
* Expanding the scope to include additional data modalities and generative model types could reveal more insights .

### Discussion Points
* The paper opens discussions on the practical implications of diffusion models in fields like data compression and creative arts .
* There's an opportunity for discourse on how to mitigate potential biases in datasets used to train such generative models. E.g. uses in medical imaging and potential risks.
* Broader impacts on technology and society, including the possible creation and proliferation of deepfakes, are worthy of further exploration .

## Paper 11: [Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502.pdf?trk=cndc-detail) <a name="paper-11"></a>

**TL;DR**: Introduces Denoising Diffusion Implicit Models (DDIMs), an efficient class of generative models that significantly accelerates the sampling process of Denoising Diffusion Probabilistic Models (DDPMs) without compromising sample quality, allowing for deterministic generation and high-quality reconstruction from latent space.

### Strengths
- **Efficiency**: Demonstrates up to 50x faster sampling compared to DDPMs, offering a significant improvement in computational efficiency.
- **Quality and Consistency**: Maintains high sample quality with much fewer generation steps and ensures consistency in generated samples when varying the sampling trajectory length.
- **Flexibility**: Provides the flexibility to trade-off between computation and sample quality effectively and supports semantically meaningful image interpolation directly in latent space.
- **Reconstruction**: Unlike DDPMs, DDIMs allow for accurate reconstruction of observations from their latent representations, highlighting its potential for broader applications beyond sample generation.

### Improvements
- **Theoretical Grounding**: While promising, the theoretical understanding of why DDIMs can efficiently reduce the number of sampling steps without losing generative performance compared to DDPMs could be further developed.
- **Diversity in Applications**: Exploring and demonstrating the efficacy of DDIMs across a wider range of datasets and domains beyond image generation can further validate its versatility.
- **Comparison with Other Models**: A more extensive comparison with other state-of-the-art generative models, including GANs and Variational Autoencoders, in terms of sample quality, efficiency, and use cases, could provide a clearer positioning of DDIMs.

### Discussion Points
- **Generative Modeling Evolution**: DDIMs challenge current generative modeling practices by emphasizing efficiency and determinism over stochasticity.
- **Broader Applications**: Exploring DDIMs in contexts beyond image generation, such as data compression and unsupervised learning, could uncover new utilities.
- **Future Research Directions**: Investigating the integration with continuous-time models like Neural ODEs could enhance DDIMs' efficiency and flexibility further.

## Paper 12: [Score-Based Generative Modeling Through Stochastic Differential Equations](https://arxiv.org/pdf/2011.13456.pdf) <a name="paper-12"></a>

**TL;DR**: Introduces a framework using stochastic differential equations (SDEs) for score-based generative modeling, providing a unified view on existing models and introducing new sampling procedures, exact likelihood computation, and controllable generation.

### Strengths
- **Unified Framework**: Offers a comprehensive framework that generalizes previous score-based and diffusion probabilistic models.
- **Efficiency and Quality**: Enables exact likelihood computation, efficient sampling, and high-fidelity image generation.
- **Controllable Generation**: Demonstrates controllable generation abilities like class-conditional generation and inpainting without re-training.

### Improvements
- **Sampling Speed**: Although improvements are noted, sampling speed remains slower compared to GANs, indicating room for efficiency enhancements.
- **Complexity and Hyperparameters**: The broad range of samplers introduced adds complexity and necessitates tuning a large number of hyperparameters.
- **Theoretical Understanding**: Deepens the theoretical understanding of SDEs in generative modeling but suggests further exploration, especially in understanding the stability and dynamics of reverse SDEs.

### Discussion Points
- **Integrating with GANs**: Investigating ways to combine the model's stable learning framework with the fast sampling capabilities of GANs.
- **Hyperparameter Optimization**: Developing methods for automatic selection and tuning of the introduced samplers' hyperparameters.
- **Broader Applicability**: Exploring the framework's potential in domains beyond image generation, such as audio synthesis or 3D shape generation.

## Paper 13: [Implicit Generation and Modeling with Energy-Based Models](https://proceedings.neurips.cc/paper/2019/file/378a063b8fdb1db941e34f4bde584c7d-Paper.pdf) <a name="paper-13"></a>

**TL;DR**: The paper introduces a scalable method for training energy-based models (EBMs) using Markov Chain Monte Carlo (MCMC) on high-dimensional data, achieving competitive performance in sample generation, inpainting, and out-of-distribution classification.

### Strengths
- EBMs generate high-fidelity images, rivaling GANs, and show no mode collapse.
- The paper demonstrates EBMs' versatility in tasks such as out-of-distribution detection, adversarial robustness, and inpainting.
- Introduces efficient training strategies for EBMs, making them applicable to complex, high-dimensional datasets.

### Improvements
- Sampling efficiency, particularly in high dimensions, could be improved to reduce computational demands.
- Extensive benchmarking across more varied datasets could solidify the model's performance claims.
- More profound theoretical insights into the working of Langevin dynamics within EBMs could enhance the interpretability of the model.

### Discussion Points
- Future work could explore latent space learning with EBMs for more complex generative tasks.
- Detailed comparisons with state-of-the-art GANs and VAEs could highlight the unique advantages and limitations of EBMs.
- Assessing the impact of EBMs on real-world problems would demonstrate their practical utility beyond academic benchmarks.

## Paper 14: [How to Train Your Energy-Based Models](https://arxiv.org/pdf/2101.03288.pdf) <a name="paper-14"></a>

**TL;DR**: This paper provides a comprehensive tutorial on training energy-based models (EBMs), detailing techniques including MCMC, Score Matching, Noise Contrastive Estimation, and outlining their theoretical connections and practical considerations.

### Strengths
- **Comprehensive Overview**: Offers a thorough and useful explanation of EBM training techniques.
- **Theoretical Depth**: Explores the theoretical foundations of EBMs and the relationships between different training methods.
- **Practical Guidance**: Provides practical advice on implementation and addresses challenges in training EBMs.
- **Broad Applicability**: Discusses a variety of applications where EBMs are advantageous due to their flexibility and expressive power.

### Improvements
- **Complexity**: The intricacy of EBM training methods may be daunting for practitioners new to the field.
- **Computational Demands**: Some discussed methods require significant computational resources, especially for high-dimensional data.
- **Implementation Challenges**: While the paper guides the reader through training EBMs, actual implementation and tuning could be non-trivial.

### Discussion Points
- **Scalability**: How can the scalability of EBM training methods be improved for real-world, high-dimensional datasets?
- **Sample Efficiency**: Investigating ways to increase sample efficiency during the training and inference of EBMs could be valuable since even the most sample-efficient EBMs are significantly slower than Generative Adversarial Networks (GANs) to generate samples. 
- **Comparison with Other Models**: A direct comparison of the performance and efficiency of EBMs against other model types would be useful.
- **Future Research Directions**: Further exploration of MCMC-free approaches to training EBMs could provide insights into more efficient algorithms.

## Paper 15: [Variational Bayesian Optimal Experimental Design](https://proceedings.neurips.cc/paper_files/paper/2019/file/d55cbf210f175f4a37916eafe6c04f0d-Paper.pdf) <a name="paper-15"></a>

**TL;DR**: The paper advances the field of Bayesian optimal experimental design by proposing variational methods for efficient estimation of expected information gain, making the design process more practical for complex, high-dimensional models.

### Strengths
- Provides a more practical and computationally efficient approach to estimate the expected information gain (EIG) for complex experimental designs.
- Establishes a strong theoretical foundation with proven convergence rates and robustness in the variational approach to EIG estimation.
- Introduces the concept of amortizing intractable computations in EIG estimation, which is novel in the context of experimental design.

### Improvements
- Some variational estimators can converge to biased estimates if the variational family does not contain the target distribution.
- The accuracy of the EIG estimation is contingent upon the choice of the variational family.
- Although improvements are made, challenges remain in models where the parameter space or outcome space is very high-dimensional.

### Discussion Points
- Future work could focus on improving scalability and reducing computational costs further, particularly for very high-dimensional parameter spaces.
- Investigating the use of richer variational families that can capture the true posterior more accurately would be beneficial.
- Explores how these variational estimators can be integrated into a real-time adaptive sequential experimental design framework, showcasing practical applications.
- Provides empirical benchmarks against existing methods, showing improvements in terms of bias and variance in estimated EIG across various designs.

## Paper 16: [On Variational Bounds of Mutual Information](https://proceedings.mlr.press/v97/poole19a/poole19a.pdf) <a name="paper-16"></a>

**TL;DR**: The paper develops a continuum of variational lower bounds for mutual information, which provides a trade-off between bias and variance, improving estimation in high-dimensional settings.

### Strengths
- Introduces a framework that integrates existing variational bounds into a continuum, improving flexibility.
- Empirical evaluations demonstrate improved bias-variance trade-offs compared to previous bounds.
- Provides a thorough theoretical grounding with proofs for new and existing bounds.
- Useful in practical applications like representation learning, where mutual information is maximized.

### Improvements
- Although improvements on bias and variance are noted, some bounds still show high variance or require large sample sizes.
- The paper primarily focuses on theoretical aspects, with limited real-world applications or datasets to demonstrate practical utility.
- Dependencies on batch sizes and specific parameter settings may limit broader applicability.
- Could explore more on the computational efficiency of implementing these bounds in practice.

### Discussion Points
- How might the new continuum of bounds impact other areas of machine learning beyond representation learning?
- What are potential ways to reduce the dependency on large sample sizes?
- Could these approaches be adapted or extended to other types of data beyond the high-dimensional types discussed?
- Discussion on the balance between theoretical optimality and practical implementation challenges in mutual information estimation.
