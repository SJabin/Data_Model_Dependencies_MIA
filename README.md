# Data and Model Dependencies of MIA
Codes and datasets for to the paper: https://arxiv.org/pdf/2002.06856.pdf

Citation: Tonni, S. M., Vatsalan, D., Farokhi, F., Kaafar, D., Lu, Z., & Tangari, G. (2020). Data and model dependencies of membership inference attack. arXiv preprint arXiv:2002.06856.

# Prerequisites

Run the script.sh file for installing all the prerequisite python packages. Required packages:

1. Python 3.5
2. Theano 1.0.5
3. Lasagne 0.2

# Running the experiments
Run the experiments on the data properties from data_properties\experiments.py and experiments on the model properties from model_properties\experiments.py. We implement all the experiments as a selected property value vs the Membership Inference Attack (MIA). For example, we sample records from the dataset according to the desired class balance and run against MIA for the experiment on the dataset's class balance. 


Simply call the ```main``` function from the ```experiments.py```. ```main``` takes two arguments - name of the dataset and name of the experiment. For example, for experimenting on different

```
from data_properties.experiments import main
# main( datalabel, exp)
main("Texas_std", "class")
```

>The value of ```exp``` can be one of the tags from ['datasize', 'class', 'feature', 'feat_no', 'entropy'] for the data properties, or from ['n_nodes','l_rates', 'l2_ratios', 'combination', 'mutual_info', 'mia_ind', 'fairness'] for the model properties.


# Experiments on the Data Properties

We explore below data properties against MIA:
1. Datasize  
2. Class balance 
3. Feature balance
4. No of fetaures
5. Entropy

# Experiments on the Model Properties

Available experiments:
1. Model architecture

* Number of hidden layers
* Number of nodes per layer
* Different learning rates
* Different l2-ratios

3. Target-shadow model combintaion

* Added target and shadow models : Artifical Neural Network (ANN), Logistic Regression (LR), Support Vector Classifier (SVC), Random Forest (RF), K-nearest Neighbors Classifier (KNN).
* Also, added a stacked shadow model (denoted as 'All') after stacking all 5 experimented models together using scikit-learn's StackingClassifier. 
4. Mutual Information between Records and Model Parameters
5. Models' Fairness: Added the experiments on the group, predictive and individual fairnesses.

7. MIA-Indistinguishability: Added the experiment on the MIA-indistinguishablity as proposed in "Disparate Vulnerability: on the Unfairness of Privacy
Attacks Against Machine Learning" (https://arxiv.org/pdf/1906.00389v1.pdf)


# Evaluating the Proposed Regularizers
(To be added)
