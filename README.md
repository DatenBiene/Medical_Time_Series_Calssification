# Medical_Time_Series_Classification

Deep Learning for Medical Time Series Classification applied to ECG examination and Lung Transplantation

***Authors:*** Pierre Delanoue, Aymeric Floyrac and Clément Gueneau

This repository contains the code and materials used to obtain the results of our study on medical time series classification.
This project was the final evaluation of the course of Deep learning for medical imaging taught by O.COLLIOT and M.VAKALOPOULOU for the MVA Master, ENS Paris Saclay.

Please  read the [report](Report.pdf) explaining the implementation and the approaches choices.
 
## Content
- A [notebook](KNN_Baseline.ipynb) implementing the KNN with DWT for each of the three datasets 
- One notebook per dataset ([ECG5000](ECG5000.ipynb), [MIT-BIH](MIT_BIH.ipynb) and [lung transplantation dataset](Transplant.ipynb) )showing the results of the deep learning methods
- A folder [interpretation](interpretation) containing the script to display 1D class activation maps and a notebook which displays the results of the NN-GP method
- A folder [models](models) containing the scripts to create the different models we used
- A folder [utils](utils) containing scripts for evaluation and accessory tools.
- A folder [data](data) containing the datasets used, apart from the MIT-BIH which is too large to be hosted on github but can be found at https://www.kaggle.com/shayanfazeli/heartbeat



