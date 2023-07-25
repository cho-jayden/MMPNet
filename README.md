# MMP Net

This repository presents official implementation of the paper, "MMP Net: A Feedforward Neural Network Model with Sequential Inputs for Representing Continuous Multistage Manufacturing Processes without Intermediate Outputs", accepted for publication in the IISE transactions journal. The paper introduces a network structure called "MMP Net", specifically designed for continuous multi-stage manufacturing processes (MMPs) without intermediate outputs. 

### Requirements 
To establish the environment for the implementation, refer to requirements.txt or use pip:
```bash
pip install -r requirements.txt
```

### Implementations
All codes in this this repository are implemented with Python and PyTorch. The implemented source codes include:
1.	models/MMPNet.py – This is the implementation of MMP Net, a feed forward neural network model specifically designed for continuous multi-stage manufacturing processes (MMPs) without intermediate outputs. 
2.	data/custom_dataset.py, data/data_loader.py – These are the source code to create data loader using PyTorch for training and testing. 
3.	exp/exp_MMP_Net.py – This is source code is used to train and test the MMP Net for the experiments demonstrated in the paper. 
4.	pic/predict_graph.py – This is source code to generates the result graphs presented in the paper. 
5.	MMPNet_main.py – This is source code to run the experiments. 
6.	script/MMPNet.sh – This is bash code to run the experiments with optimal parameters. 
