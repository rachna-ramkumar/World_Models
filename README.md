# Report

## Rachna Ramkumar
   RR3656

## Task1 

Read the paper and wrote a report of the technique (a model-based RL) which can be found in this repository, in a tutorial like fashion with complete instructions to recreate the Worldmodels.

## Task 2

For this task the results have been reproduced for the car racing environment. Adopted the repo https://github.com/pantelis-classes/world-models-latest.git. All the weight files, training history files and controller history files can be found in the respective directories.

## Task 3

The scripts used for the GAN files can be found in this repository. Based on the results, we can come to the following conclusion. In principle, by combining a variational autoencoder and a generative adversarial network, the VAE reconstruction objective can be based on learned feature representations in the GAN method. As a result, we will replace element-wise errors with feature-wise errors to better capture the data distribution while providing invariance. We apply our approach to world models in terms of visual fidelity and show that it outperforms VAEs with element-wise similarity tests. Furthermore, we will demonstrate that the method learns an embedding in which high-level abstract visual features can be modified using simple arithmetic. Also, we must keep in mind that the GAN method needs more training for a better recreation of images.


# WorldModels
An implementation of the ideas from this paper https://arxiv.org/pdf/1803.10122.pdf

Code base adapted from https://github.com/hardmaru/estool

For full installation and run instructions see this blog post:

https://applied-data.science/blog/hallucinogenic-deep-reinforcement-learning-using-python-and-keras


# Building with Docker

See `Dockerfile.cpu` and `launch-docker-cpu.sh` or equivalent GPU files for instructions.


## References :

1. https://github.com/pantelis-classes/world-models-latest
2. https://arxiv.org/pdf/1803.10122.pdf
3. https://github.com/hardmaru/estool
