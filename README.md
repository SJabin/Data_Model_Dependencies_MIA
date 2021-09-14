# Data_Model_Dependencies_MIA
Codes and datasets for to the paper: https://arxiv.org/pdf/2002.06856.pdf

Citation: Tonni, S. M., Vatsalan, D., Farokhi, F., Kaafar, D., Lu, Z., & Tangari, G. (2020). Data and model dependencies of membership inference attack. arXiv preprint arXiv:2002.06856.

# Prerequisites

Run the script.sh file for installing all the prerequisite python packages.

# Experiments on the Data Properties

To run experiments go to the data_properties\experiments.py.

Available experiments:
1. Datasizes vs MIA
2. Class balance vs MIA
3. Feature balance vs MIA
4. No of fetaures vs MIA
5. Entropy vs MIA

# Experiments on the Model Properties

To run experiments go to the model_properties\experiments.py.

Available experiments:
1. Model architecture

* Number of hidden layers
* Number of nodes per layer
* Different learning rates
* Different l2-ratios

3. Target-shadow model combintaion

* Added target and shadow models : Artifical Neural Network (ANN), Logistic Regression (LR), Support Vector Classifier (SVC), Random Forest (RF), K-nearest Neighbors Classifier (KNN).
* Also, added a stacked shadow model ('All') after stacking all 5 experimented models together using scikit-learn's StackingClassifier. 

5. Fairness [ TO DO ]
6. MIA-Indistinguishability [ TO DO ] 

# Evaluating the Regularizers
