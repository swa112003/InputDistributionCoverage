# IDC
This repository contains the official implementation of the IDC Framework proposed in our paper "Input Distribution Coverage: Measuring Feature Interaction
Adequacy in Neural Network Testing".

## Datasets
- [dSprites](https://github.com/deepmind/dsprites-dataset)
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Components
1. Out-of-distribution data detection 
2. Variational Autoencoder
3. Combinatorial Coverage Measurement

### Out-of-distribution data detection 
We used the Likelihood-Regret<sup>[[1]](#1)</sup> based out-of-distribution (OOD) filter in the framework. [Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder](https://github.com/XavierXiao/Likelihood-Regret) has the code and pretrained models available.

### Variational Autoencoder
Variational Autoencoder (VAE) models from two repositories are used in this work. 
- FactoVAE<sup>[[2]](#2)</sup> and $\beta$-TCVAE<sup>[[3]](#3)</sup> from [https://github.com/YannDubs/disentangling-vae](https://github.com/YannDubs/disentangling-vae)
- TwoStageVAE<sup>[[4]](#4)</sup> from [https://github.com/daib13/TwoStageVAE](https://github.com/daib13/TwoStageVAE)

### Combinatorial Coverage Measurement
Combinatorial Coverage Measurement (CCM) Command Line Tool from [https://github.com/usnistgov/combinatorial-testing-tools](https://github.com/usnistgov/combinatorial-testing-tools)


## References
<a id="1">[1]</a> Xiao, Zhisheng, Qing Yan, and Yali Amit. "Likelihood regret: An out-of-distribution detection score for variational auto-encoder." Advances in neural information processing systems 33 (2020): 20685-20696.

<a id="2">[2]</a> Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising." International Conference on Machine Learning. PMLR, 2018.

<a id="3">[3]</a> Chen, Ricky TQ, et al. "Isolating sources of disentanglement in variational autoencoders." Advances in neural information processing systems 31 (2018).

<a id="4">[4]</a> Dai, B. and Wipf, D. Diagnosing and enhancing VAE models. In International Conference on Learning Representations, 2019.
