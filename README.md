# MMP Net

This repository presents official implementation of a paper, "MMP Net: A Feedforward Neural Network Model with Sequential Inputs for Representing Continuous Multistage Manufacturing Processes without Intermediate Outputs", which has been accepted for publication in the IISE transactions journal. The paper introduced a network structure called "MMP Net", specifically designed for continuous multi-stage manufacturing (MMPs) without intermediate outputs. 

### Requirements 
To build the environment of the implementation, check requirements.txt or use pip:
'''bash
pip install -r requirements.txt
'''

### Implementations
All codes of this repository are implemented with Python and PyTorch. The implemented source codes are as followed:
1.	models/MMPNet.py – This is Implementation of MMP Net, a feed forward neural network model specifically designed for continuous multi-stage manufacturing processes (MMPs) without intermediate outputs. 
2.	data/custom_dataset.py, data/data_loader.py – These are source code to make data loader of pytorch for training and testing. 
3.	exp/exp_MMP_Net.py – This is source code to train and test the MMP Net for the experiments in the paper. 
4.	pic/predict_graph.py – This is source code to print the result graphs in the paper. 
5.	MMPNet_main.py – This is source code to run the experiments. 
6.	script/MMPNet.sh – This is bash code to run the experiments with optimal parameters. 

