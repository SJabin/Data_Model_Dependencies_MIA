from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

import os
import imp
import time
import random
import numpy as np
import pandas as pd
import lasagne
import theano

n_shadow=1
theano.config.optimizer='fast_compile'
np.random.seed(50215)


from utils.data_utils import get_deafults
from models.attack import train_target_model, train_shadow_model, train_attack

#sampling records 

    
def mia_exec(target_data, shadow_data, shadow_all_indices):
    #train target
    attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data)

    #train shadow
    shadow_dataset=shadow_data, shadow_all_indices
    attack_train_x, attack_train_y, train_classes = train_shadow_model(shadow_dataset)
            
    #train attack
    attack_datasets=attack_train_x, attack_train_y, attack_test_x, attack_test_y
    classes=train_classes, test_classes                
    atk_results=train_attack(attack_datasets,classes)
    
    return target_results, atk_results        

    

def get_indices(trsize, shsize, data, exp, cbal=None, fbal=None, feat_no=None, n_shadow=1):
    # For experiment on the class or feature balance
    # first split the dataset into two parts. 
    # class balance is measured on class=1.
    # feature balance is measured over first 5 features.
    # balance is measured on the group of records that has the feature value "<=.5" for each of the five features.
    if exp == 'class' or exp == 'feature':
        if exp == 'class':
            _size= int(cbal* trsize / 100)
#             print(_size)
    
            dt0=data[data['class']==0].reset_index(drop=True)
            dt1=data[data['class']==1].reset_index(drop=True)
        
        if exp == 'feature':
            _size= int(fbal* trsize / 100)

            
            dt1=data
            for i in range(0, 5):
                dt1=dt1[dt1.iloc[:,i]<=.5]
            temp_ind=np.setdiff1d(list(data.index), list(dt1.index))
            dt0= data.iloc[temp_ind,:].reset_index(drop=True)
            dt1=dt1.reset_index(drop=True)

    
        indices1=np.arange(len(dt1))
        indices0=np.arange(len(dt0))      
        
        #sample _size records from class = 1  | features <=.5
        np.random.shuffle(indices1)
        target_indices = random.sample(indices1.tolist(),_size)
        shadow_indices = np.setdiff1d(indices1, target_indices)
        target_d1=dt1.iloc[target_indices,:].reset_index(drop=True)
        shadow_d1=dt1.iloc[shadow_indices,:].reset_index(drop=True)
        
        #sample rest of the records from class = 0 | features >.5
        np.random.shuffle(indices0)        
        target_indices = random.sample(indices0.tolist(),trsize-_size)
        shadow_indices = np.setdiff1d(indices0, target_indices)    
        target_d0=dt0.iloc[target_indices,:].reset_index(drop=True)
        shadow_d0=dt0.iloc[shadow_indices,:].reset_index(drop=True)
                    
        target_data=pd.concat([target_d0,target_d1]).reset_index(drop=True)       
        tgt_ind=list(target_data.index)
        np.random.shuffle(tgt_ind)
        target_data = target_data.iloc[tgt_ind, :].reset_index(drop=True)

        
        shadow_data=pd.concat([shadow_d0,shadow_d1]).reset_index(drop=True)
        shd_ind=list(shadow_data.index)
        np.random.shuffle(shd_ind)
        shadow_data = shadow_data.iloc[shd_ind, :].reset_index(drop=True)
        
    else:
        if exp == 'feat_no' or exp == 'entropy':
            labels = data['class']
            data = data.iloc[:, 0:feat_no]
            data['class']=labels
        indices=np.arange(len(data))
        target_indices = random.sample(indices.tolist(),trsize)
        shadow_indices = np.setdiff1d(indices, target_indices)
    
        target_data=data.iloc[target_indices,:].reset_index(drop=True)
        shadow_data=data.iloc[shadow_indices,:].reset_index(drop=True)
    
    #indices for multiple shadows.
    shadow_all_indices=[]
    temp=np.arange(len(shadow_data))
    for j in range(0,n_shadow): shadow_all_indices.append(random.sample(temp.tolist(),shsize)) 
        
    return target_data, shadow_data, shadow_all_indices
        
def exp_datasize(data, trsize, shsize):    
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exp='datasize')
        defaults = get_deafults(target_data)
        target, atk = mia_exec(target_data, shadow_data, shadow_all_indices)
        return trsize, shsize, defaults, target, atk
    
    
# def exp_shadowsize():    
#     for shsize in range (minsize, maxsize+1, step):
#         trsize=minsize
#         target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='datasize')
#         defaults= get_deafults(target_data)
#         target, atk = mia_exec(target_data, shadow_data, shadow_all_indices)
#         return trsize, shsize, defaults, target, atk

def exp_classbalance(data, trsize, shsize, cbal):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exp='class', cbal = cbal)               
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices)
    return trsize, shsize, defaults, target, atk

def exp_featurebalance(data, trsize, shsize, fbal):

    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exp='feature', fbal = fbal) 
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices)
    return trsize, shsize, defaults, target, atk
        
def exp_featno(data,  trsize, shsize, featsize):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exp='feat_no', feat_no = featsize)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices)
    return trsize, shsize, defaults, target, atk

def exp_entropy(data,  trsize, shsize, featsize):
    #measured entropy over datasets with different feature size
    return exp_featno(data,  trsize, shsize, featsize)


def save_results(itr, feat_no, result, savefile):
    trsize, shsize, defaults, target, atk = result
    
    fpr,tpr,roc_auc,ts_prec,ts_rec, ts_fbeta, tr_acc, ts_acc=target
    atk_prec, atk_rec, atk_acc, class_acc=atk#, c_atk_acc=result
    
    result = str(itr)+','+str(trsize)+','+str(shsize)+','+ str(feat_no)+ ','
    result += str(ts_prec)+','+str(ts_rec)+','+str(tr_acc)+','+str(ts_acc)+','+str(roc_auc)+','

#                 for j in range(0, len(fpr)):
#                     result+='fpr'+str(fpr[j])+','
#                 for j in range(0, len(tpr)):
#                     result+='tpr'+str(tpr[j])+','
            
    result+=str(atk_prec)+','+str(atk_rec)+','+str(atk_acc)#+','+ str(class_acc[0])+','+ str(class_acc[0])

    text_file=open(savefile, 'a')
    text_file.write(result)
    text_file.write('\n')
    text_file.close()
    
    
    
    

#exp = ['datasize', 'class', 'feature', 'feat_no', 'entropy']


def main(datalabel, exp):
    
    # for the datasize experiment datasizes varied between [1000, 10000] for Adult dataset
    # for Purchase and Texas datasizes varied between [10000, 100000]
    # change the minsize = 10000, maxsize = 100000, step=10000 for Purchase and Texas
    # for all other experiments datasize is fixed. For Adult, datasize =10000
    # for Purchase and Texas change datasize = 100000
    
    
    filename='./'+datalabel+'.csv'
    savefile=exp+'_'+datalabel+'.csv'
    data=pd.read_csv(filename)
    feat_no=len(data.columns) - 1
    
    for itr in range(0,2):
        print("\nIteration: ", itr)
        
        if exp == 'datasize':
            
            print("\nExp: Datasize============")
            minsize =1000
            maxsize = 10000
            step=5000
            
            print("\nVarying target size\n")
            for trsize in range (minsize, maxsize+1, step):
                shsize=minsize
                print ("target: ", trsize, ", shadow: ", shsize)
                result = exp_datasize(data, trsize, shsize)
                save_results(itr, feat_no, result, savefile)
            
            print("\nVarying shadow size\n")
            for shsize in range (minsize, maxsize+1, step):
                trsize=minsize
                print ("target: ", trsize, ", shadow: ", shsize)
                result = exp_datasize(data, trsize, shsize)
                save_results(itr, feat_no, result, savefile)
        
        elif exp == 'class':
            print("Exp: Class Balance============")
            trsize = shsize = 10000
            for cbal in range (10, 91, 10):
                result = exp_classbalance(data, trsize, shsize, cbal)
                save_results(itr, feat_no, result, savefile)
            
        elif exp == 'feature':
            print("Exp: Feature Balance============")
            trsize = shsize = 10000
            for fbal in range (10, 91, 10):
                result = exp_featurebalance(data, trsize, shsize, fbal)
                save_results(itr, feat_no, result, savefile)

            
        elif exp == 'feat_no':
            print("Exp: No of Features============")
            trsize = shsize = 10000
            for fsize in range (1, feat_no, 1):
                print(fsize)
                result = exp_featno(data,  trsize, shsize, fsize)
                save_results(itr, fsize, result, savefile)
            
        elif exp == 'entropy':
            print("Exp: Entropy============")
            trsize = shsize = 10000
            for fsize in range (1, feat_no, 1):
                result = exp_entropy(data,  trsize, shsize, fsize)
                save_results(itr, fsize, result, savefile)
        
               
            
    
   
    

if __name__ == '__main__':

    main("")