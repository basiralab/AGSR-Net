"""Main function of Adversarial Graph Super-Resolution Network (AGSR-Net) framework for predicting high-resolution
    connectomes from low-resolution connectomes.
    ---------------------------------------------------------------------

    This file contains the implementation of the training and testing process of our AGSR-Net model.
        train(model, subjects_adj, subjects_ground_truth, args)
                Inputs:
                        model:        constructor of our AGSR-Net model:  model = AGSRNet(ks,args)
                                      ks:   array that stores reduction rates of nodes in Graph U-Net pooling layers
                                      args: parsed command line arguments
                        subjects_adj: (n × l x l) tensor stacking LR connectivity matrices of all training subjects
                                       n: the total number of training subjects
                                       l: the dimensions of the LR connectivity matrices
                        subjects_ground_truth: (n × h x h) tensor stacking LR connectivity matrices of all training subjects
                                                n: the total number of training subjects
                                                h: the dimensions of the LR connectivity matrices
                        args:          parsed command line arguments, to learn more about the arguments run:
                                       python demo.py --help
                Output:
                        for each epoch, prints out the mean training MSE error

        test(model, test_adj,test_ground_truth,args)
                Inputs:
                        test_adj:      (t × l x l) tensor stacking LR connectivity matrices of all testing subjects
                                        t: the total number of testing subjects
                                        l: the dimensions of the LR connectivity matrices
                        test_ground_truth:      (t × h x h) tensor stacking LR connectivity matrices of all testing subjects
                                                 t: the total number of testing subjects
                                                 h: the dimensions of the LR connectivity matrices
                        see train method above for model and args.
                Outputs:
                        for each epoch, prints out the mean testing MSE error
    ---------------------------------------------------------------------
    Copyright 2020 Megi Isallari, Istanbul Technical University.
    All rights reserved.
    """

from preprocessing import *
from model import *
from train import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AGSR-Net')
    parser.add_argument('--epochs', type=int, default=200, metavar='no_epochs',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='lr',
                        help='learning rate (default: 0.0001 using Adam Optimizer)')
    parser.add_argument('--lmbda', type=float, default=0.1, metavar='L',
                        help='self-reconstruction error hyperparameter')
    parser.add_argument('--lr_dim', type=int, default=160, metavar='N',
                        help='adjacency matrix input dimensions')
    parser.add_argument('--hr_dim', type=int, default=320, metavar='N',
                        help='super-resolved adjacency matrix output dimensions')
    parser.add_argument('--hidden_dim', type=int, default=320, metavar='N',
                        help='hidden GraphConvolutional layer dimensions')
    parser.add_argument('--padding', type=int, default=26, metavar='padding',
                        help='dimensions of padding')
    parser.add_argument('--mean_dense', type=float, default=0., metavar='mean',
                        help='mean of the normal distribution in Dense Layer')
    parser.add_argument('--std_dense', type=float, default=0.01, metavar='std',
                        help='standard deviation of the normal distribution in Dense Layer')
    parser.add_argument('--mean_gaussian', type=float, default=0., metavar='mean',
                        help='mean of the normal distribution in Gaussian Noise Layer')
    parser.add_argument('--std_gaussian', type=float, default=0.1, metavar='std',
                        help='standard deviation of the normal distribution in Gaussian Noise Layer')
    args = parser.parse_args()


subjects_adj, subjects_ground_truth, test_adj, test_ground_truth = data()

ks = [0.9, 0.7, 0.6, 0.5]
model = AGSRNet(ks, args)

print(model)

# SIMULATING THE DATA: EDIT TO ENTER YOUR OWN DATA
subjects_adj = np.random.normal(0.5, 1, (190, 160, 160))
test_adj = np.random.normal(0.5, 1, (87, 160, 160))
subjects_ground_truth = np.random.normal(0.5, 1, (190, 268, 268))
test_ground_truth = np.random.normal(0.5, 1, (87, 268, 268))

train(model, subjects_adj, subjects_ground_truth, args)
test(model, test_adj, test_ground_truth, args)
