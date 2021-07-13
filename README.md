# AGSR-Net: Adversarial Graph Super-Resolution Network

This repository provides the official PyTorch implementation of the following paper:

**Brain Graph Super-Resolution Using Adversarial Graph Neural Network with Application to Functional Brain Connectivity.**

[Megi Isallari](https://github.com/meg-i)<sup>1</sup>, [Islem Rekik](https://basira-lab.com/)<sup>1</sup>

> <sup>1</sup>BASIRA Lab, Faculty of Computer and Informatics Engineering, Istanbul Technical University, Istanbul, Turkey

Please contact isallari.megi@gmail.com for further inquiries. Thanks.

Brain image analysis has advanced substantially in recent years with the proliferation of neuroimaging datasets acquired at different resolutions. While research on brain image super-resolution has undergone a rapid development in the recent years, brain graph super-resolution is still poorly investigated because of the complex nature of non-Euclidean data. In this paper, we propose the first-ever deep graph super-resolution (GSR) framework that attempts to automatically generate high-resolution (HR) brain graphs with _N'_ nodes (i.e, anatomical regions of interest (ROIs)) from low-resolution (LR) graphs with _N_ nodes where _N < N'_. First, we formalize our GSR problem as a node feature embedding learning task. Once the HR nodes’ embeddings are learned, the pairwise connectivity strength between brain ROIs can be derived through an aggregation rule based on a novel Graph U-Net architecture. While typically the Graph U-Net is a node-focused architecture where graph embedding depends mainly on node attributes, we propose a graph-focused architecture where the node feature embedding is based on the graph topology. Second, inspired by graph spectral theory, we break the
symmetry of the U-Net architecture by super-resolving the low-resolution brain graph structure and node content with a GSR layer and two graph convolutional network layers to further learn the node embeddings in the
HR graph. Third, to handle the domain shift between the ground-truth and the predicted HR brain graphs, we incorporate adversarial regularization to align their respective distributions. Our proposed AGSR-Net framework outperformed its variants for predicting high-resolution functional brain graphs from low-resolution ones.

![AGSR-Net pipeline](/images/concept_fig.png)

# Detailed proposed AGSR-Net pipeline

This work has been accepted for publication by “Medical Image Analysis” (MedIA) journal. The key idea of AGSR-Net can be summarized in four fundamental steps: (i) learning feature embeddings for each brain ROI
(node) in the LR connectome, (ii) the design of a graph super-resolution operation that predicts an HR connectome from the LR connectivity matrix and feature embeddings of the LR connectome computed in (i), (iii) learning node feature embeddings for each node in the super-resolved (HR) graph obtained in (ii), (iv) integrating an adversarial model that acts as a discriminator to distinguish whether a HR connectome is from a prior ground-truth HR distribution or the generated HR connectome in (iii). We evaluated our framework on 277 subjects from the Southwest University Longitudinal Imaging Multimodal (SLIM) study:

http://fcon_1000.projects.nitrc.org/indi/retro/southwestuni_qiu_index.html.

In this repository, we release the code for training and testing AGSR-Net on the SLIM dataset.

![AGSR-Net pipeline](/images/overallfig.png)

# Dependencies

The code has been tested with Google Colaboratory which uses Ubuntu 18.04.3 LTS Bionic,
Python 3.6.9 and PyTorch 1.4.0. In case you opt to run the code locally, you need to install the following python packages via pip:

- [Python 3.6+](https://www.python.org/)
- [PyTorch 1.4.0+](http://pytorch.org/)
- [Scikit-learn 0.22.2+](https://scikit-learn.org/stable/)
- [Matplotlib 3.2.2+](https://matplotlib.org/)
- [Numpy 1.18.5+](https://numpy.org/)

# Demo

We provide a demo code in `demo.py` to run the script of AGSR-Net for predicting high-resolution connectomes from low-resolution functional brain connectomes. To set the parameters, you should provide commandline arguments.

If you want to run the code in the hyperparameters described in the paper, you can run it without any commandline arguments:

```sh
$ python demo.py
```

It would be equivalent to:

```sh
$ python demo.py --epochs 200 --lr 0.0001 --lmbda 0.1 --lr_dim 160 --hr_dim 320 --hidden_dim 320 --padding 26 --mean_dense 0 --std_dense 0.01 --mean_gaussian 0 --std_gaussian 0.1
```

To learn more about how to use the arguments:

```sh
$ python demo.py --help
```

| Plugin        | README                                            |
| ------------- | ------------------------------------------------- |
| epochs        | number of epochs to train                         |
| lr            | learning rate of Adam Optimizer                   |
| lmbda         | self-reconstruction error hyper-parameter         |
| lr_dim        | number of nodes of low-resolution brain graph     |
| hr_dim        | number of nodes of high-resolution brain graph    |
| hidden_dim    | number of hidden GCN layer neurons                |
| padding       | dimensions of padding                             |
| mean_dense    | mean of the normal distribution in Dense Layer    |
| std_dense     | std of the normal distribution in Dense Layer     |
| mean_gaussian | mean of the normal distribution in Gaussian Layer |
| std_gaussian  | std of the normal distribution in Gaussian Layer  |

We also note that in our published paper, K was set to 2.

**Data preparation**

In our paper, we have used the SLIM dataset. In this repository, we simulated a n x l x l tensor and a t x l x l (low-resolution connectomes for training and testing subjects respectively where l is the number of nodes of the LR connectome) as well as a n x h x h tensor and t x h x h (high-resolution connectomes for training and testing subjects respectively where h is the number of nodes of the HR connectome). It might yield in suboptimal results since data is randomly generated opposed to real brain graph data.

To use a dataset of your own preference, you can edit the data() function at preprocessing.py. In order to train and test the framework, you need to provide:

1. `N` low-resolution brain graph connectomes of dimensions `L*L` for variable `subjects_adj` in `demo.py`
2. `N` high-resolution brain graph connectomes of dimensions `H*H` for variable `subjects_ground_truth` in `demo.py`
3. `T` low-resolution brain graph connectomes of dimensions `L*L` for variable `test_adj` in `demo.py`
4. `T` high-resolution brain graph connectomes of dimensions `H*H` for variable `test_ground_truth` in `demo.py`

# Example Results

If you run the demo with the default parameter setting as in the command below,

```sh
$ python demo.py –epochs=200 –lr=0.0001 –splits=5 –lmbda=16 –lr_dim=160 –hr_dim=320 –hidden_dim=320 –padding=26
```

you will get the following outputs:

![AGSR-Net pipeline](/images/example.jpg)

# Related references

Graph U-Nets: Gao, H., Ji, S.: Graph u-nets. In Chaudhuri, K., Salakhutdinov, R., eds.: Proceedings of the
36th International Conference on Machine Learning. Volume 97 of Proceedings of Machine Learning Research., Long Beach, California, USA, PMLR (2019) 2083–2092 [https://github.com/HongyangGao/Graph-U-Nets]

SLIM Dataset: Liu, W., Wei, D., Chen, Q., Yang, W., Meng, J., Wu, G., Bi, T., Zhang, Q., Zuo, X.N., Qiu,
J.: Longitudinal test-retest neuroimaging data from healthy young adults in southwest china. Scientific Data 4 (2017) [https://www.nature.com/articles/sdata201717]

# AGSR-Net paper on arXiv:

Coming up soon.

# Please cite the following paper when using AGSR-Net:

```latex
@article{isallari2021brain,
  title={Brain graph super-resolution using adversarial graph neural network with application to functional brain connectivity},
  author={Isallari, Megi and Rekik, Islem},
  journal={Medical Image Analysis},
  volume={71},
  pages={102084},
  year={2021},
  publisher={Elsevier}
}
```
