# Explainable Shape Analysis through Deep Hierarchical Generative Models

The Tensorflow code in this repository implements the modifications of the VAE and Ladder VAE frameworks presented in *[Learning interpretable anatomical features through deep generative models: Application to cardiac remodeling](https://arxiv.org/pdf/1807.06843.pdf)* and Explainable Shape Analysis through Deep Hierarchical Generative Models: Application to Cardiac Remodelling papers.

### Usage

Make sure you have *[Python 3.4](https://www.python.org/downloads/windows/)* and *[Tensorflow](https://www.tensorflow.org/install/)* installed.

The architecture and training details can be configured in the `config/config.json` file.

To trainin the network please run:

`python training.py --config=jsons/config.json` 


***

### Acknowledgments

This implementation was inspired by *[geosada](https://github.com/geosada/LVAE)* Tensorflow implementation of the LVAE original paper. If you find this work useful, please cite the following papers: 

[1] Biffi, C., et al. *[Learning interpretable anatomical features through deep generative models: Application to cardiac remodeling](https://arxiv.org/pdf/1807.06843.pdf)* International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.

[2] Biffi, C., et al. *[Explainable Shape Analysis through Deep Hierarchical Generative Models: Application to Cardiac Remodelling](https://arxiv.org/pdf/1807.06843.pdf)*
