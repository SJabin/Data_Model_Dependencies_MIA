import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

import theano.tensor as T
import lasagne
import theano

from models.classifier import train_ANN 
from utils.model_utils import get_fairness, get_membership_indistinguishability


#------------------------------training------------------------------

def train_target_model(data, save_params=None, n_layer=1, n_hidden = 50, lrate = .001, l2ratio = 1e-7, model ='ANN'):    
    tr_acc=0
    label= data.iloc[:,len(data.columns)-1]
    feat = data.iloc[:,0:len(data.columns)-1]
    
    train_x, test_x, train_y,  test_y = train_test_split(feat, label, test_size=0.25, random_state=42)
    dataset=train_x.reset_index(drop=True), train_y.reset_index(drop=True), test_x.reset_index(drop=True), test_y.reset_index(drop=True)
    
    # target - attack model combinations
    if model=='LR':
        result = train_LR(dataset, epochs=50, batch_size=500, learning_rate=lrate, l2_ratio=l2ratio, n_hidden=n_hidden, n_layer=n_layer, target=True, save_params=save_params)
    if model=='SVC':
    if model=='RF':
    if model=='KNN':
    else:   
        result = train_ANN(dataset, epochs=50, batch_size=500, learning_rate=lrate, l2_ratio=l2ratio, n_hidden=n_hidden, n_layer=n_layer, target=True, save_params=save_params)
    
    if save_params != None:
        output_layer, train_pred_y, test_pred_y, avg_mi, max_mi = result
    else:
        output_layer, train_pred_y, test_pred_y = result
    
    tr_acc=accuracy_score(train_y, train_pred_y)
    ts_acc=accuracy_score(test_y, test_pred_y)
    print('target accuracies:',tr_acc, ts_acc)

    
    attack_x, attack_y = [], []
    input_var = T.matrix('x')
    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
    prob_fn = theano.function([input_var], prob)
    
    y_probs=prob_fn(test_x)
    y_probs=y_probs[:,1]
    
    all_fpr, all_tpr, all_thresholds = roc_curve(test_y, y_probs, drop_intermediate=True)
    fpr=[]
    tpr=[]
    for i in range(0, len(all_thresholds)):
        if all_thresholds[i]>=.5:
            fpr.append(all_fpr[i])
            tpr.append(all_tpr[i])
    roc_auc = roc_auc_score(test_y, y_probs)
    
    #_TP=TP(test_y,test_pred_y)
    #_FN=FN(test_y,test_pred_y)
    #_FP=FP(test_y,test_pred_y)
    #_TN=TN(test_y,test_pred_y)
    
    prec, rec, f_beta, _ = precision_recall_fscore_support(test_y, test_pred_y, average='binary')
    
    if save_params != None: 
        res = fpr,tpr, roc_auc, prec, rec, f_beta, tr_acc, ts_acc, avg_mi, max_mi
    else:
        res = fpr,tpr, roc_auc, prec, rec, f_beta, tr_acc, ts_acc
    
    attack_x=np.vstack((prob_fn(train_x), prob_fn(test_x)))
    _in=np.ones(len(train_x)).reshape(-1,1)
    _out=np.zeros(len(test_x)).reshape(-1,1)
    attack_y=np.ravel(np.vstack((_in,_out)))
    classes = np.concatenate([train_y, test_y])
    
    indices=np.arange(len(attack_x))
    np.random.shuffle(indices)
    attack_x=attack_x[indices,:]
    attack_y=attack_y[indices]
    classes=classes[indices]
        
    return attack_x, attack_y, classes, res  



def train_shadow_model(datasets):
    
    data,shadow_all_indices=datasets
    input_var = T.matrix('x')
    attack_x, attack_y = [], []
    classes = []
    for i in range(0,len(shadow_all_indices)):
        
        dt=data.iloc[shadow_all_indices[i],:].reset_index(drop=True)
        shadow_Y=dt.iloc[:,len(dt.columns)-1]
        shadow_X=dt.iloc[:,0:len(dt.columns)-1]
        #print('shadow:',np.unique(shadow_Y, return_counts=True))
        
        train_x, test_x, train_y, test_y = train_test_split(shadow_X, shadow_Y, test_size=.25, stratify=shadow_Y)
        data=train_x, train_y, test_x, test_y
        output_layer, train_pred_y, test_pred_y= train_ANN(data, epochs=10, batch_size=100, learning_rate=.001, l2_ratio=1e-7, n_hidden=50) #.01
        prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
        prob_fn = theano.function([input_var], prob)
        
        tr_score=accuracy_score(train_y, train_pred_y)
        ts_score=accuracy_score(test_y, test_pred_y)
        print('shadow accuracy:', tr_score, ts_score)
                
        attack_x=np.vstack((prob_fn(train_x), prob_fn(test_x)))
        _in=np.ones(len(train_x)).reshape(-1,1)
        _out=np.zeros(len(test_x)).reshape(-1,1)
        attack_y=np.ravel(np.vstack((_in,_out)))
        classes = np.concatenate([train_y, test_y])
    
    indices=np.arange(len(attack_x))
    np.random.shuffle(indices)
    attack_x=attack_x[indices,:]
    attack_y=attack_y[indices]   
    classes=classes[indices]
    
    return attack_x, attack_y, classes
    

def train_attack(dataset, classes, model='ANN'):
    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    
    c_score=[]
    true_y = []
    pred_y = []
    for c in unique_classes:
        #print ('---------------Training attack model for class {}-----------------'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        c_dataset = (c_train_x, c_train_y, c_test_x[:,0:2], c_test_y)
        
        if model=='ANN':
            c_train_pred_y, c_test_pred_y= train_ANN(c_dataset, epochs=50, batch_size=500, learning_rate=.0001, l2_ratio=1e-7, n_hidden=5, rtn_layer=False)
    
        true_y.append(c_test_y)
        pred_y.append(c_test_pred_y)
        
        c_score.append(accuracy_score(c_test_y, c_test_pred_y))
    
    #print ('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    
    score=accuracy_score(true_y, pred_y)
    prec, rec, _, _ = precision_recall_fscore_support(true_y, pred_y, average='binary')
    
    print('attack accuracies:',prec, rec,score)
    print('\n')
    
    res=prec,rec, score, c_score
    return res