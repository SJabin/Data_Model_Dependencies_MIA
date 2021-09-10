import pandas as pd
import numpy as np



def get_deafults(data):
    fetures = data.iloc[:,0:len(data.columns)-1]
    #entropy
    entr=get_entropy(features)
    #class_balance
    cbal, cindices = get_class_balance(data['class'])
    #feat_balance
    fbal, findices = get_feat_balance(target_data)
    
    def_mesaures = entr, cbal, cindices, fbal, findices 
    
    return def_measures

def get_entropy(features):
    base = 2
    en_x=0
    for i in range(0, len(features.columns)):
        value,counts = np.unique(features.iloc[:,i], return_counts=True)
        norm_counts = counts / counts.sum()
        en_x+=-(norm_counts * np.log(norm_counts)/np.log(base)).sum()
    en_x/=len(features.columns)
    return en_x

def get_class_balance(labels):
    classes, counts = np.unique(labels, return_counts=True)
    indices=labels[labels==1].index.tolist()
    balance = int(counts[1]/len(labels)*100)
    result=balance, indices
    assert len(indices)==counts[1]
    print('--class balance ', balance)
    return result
    

def get_feat_balance(data):
    temp2=data
    for i in range(0,5): 
        temp2=temp2[temp2['col'+str(i)]<=.5]
        
    indices=list(temp2.index)
    temp2= temp2.reset_index(drop=True)
    balance = int(len(temp2)/ len(data) *100)
    print('--feature balance ', balance)
    result = balance, indices
    
    return result
