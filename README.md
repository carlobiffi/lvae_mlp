# Explainable Shape Analysis through Deep Generative Models [![DOI](https://zenodo.org/badge/192420491.svg)](https://zenodo.org/badge/latestdoi/192420491)


The Tensorflow code in this repository implements the modifications of the VAE and Ladder VAE frameworks presented in *[Learning Interpretable Anatomical Features through Deep Generative Models: Application to Cardiac Remodeling](https://arxiv.org/abs/1807.06843)* and *[Explainable Shape Analysis through Deep Hierarchical Generative Models: Application to Cardiac Remodelling](http://arxiv.org/abs/1907.00058)* papers.

### Usage

Make sure you have *[Python 3.4](https://www.python.org/downloads/windows/)* and *[Tensorflow](https://www.tensorflow.org/install/)* installed.

The architecture and training details can be configured in the `config/config.json` file.

To train the network please run:

`python training.py --config=jsons/config.json` 


***

### Acknowledgments

This implementation was inspired by *[geosada](https://github.com/geosada/LVAE)* Tensorflow implementation of the LVAE original paper. If you find this work useful, please cite the following papers: 

[1] Biffi, C., et al. *[Explainable Shape Analysis through Deep Hierarchical Generative Models: Application to Cardiac Remodelling](http://arxiv.org/abs/1907.00058)* Submitted for review to IEEE Transactions on Medical Imaging.

[2] Biffi, C., et al. *[Learning Interpretable Anatomical Features through Deep Generative Models: Application to Cardiac Remodeling](https://arxiv.org/abs/1807.06843)* International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018.
