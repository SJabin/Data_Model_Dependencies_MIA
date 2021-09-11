#from classifier import train as train_model, iterate_minibatches
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from scipy.stats import entropy
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import math
import os
import imp
import time
import random
import theano.tensor as T
import numpy as np
import pandas as pd
import lasagne
import theano
import math
n_shadow=1
theano.config.optimizer='fast_compile'
np.random.seed(50215)

#sampling records 

    
def mia_exec(feat_no):
    #train target
    #save_params=st, savefile
    attack_test_x, attack_test_y, test_classes, train_results= train_target_model(target_data, datalabel, save_params)

    #train shadow
    shadow_dataset=shadow_data, shadow_all_indices
    attack_train_x, attack_train_y, train_classes = train_shadow_model(shadow_dataset)
            
    #train attack
    attack_datasets=attack_train_x, attack_train_y, attack_test_x, attack_test_y
    classes=train_classes, test_classes                
    atk_results=train_attack(attack_datasets,classes, target_class_indices)
    
    return target_results, atk_results        

    

def gen_indices(trsize, shsize, data, exp, cbal, fbal, feat_no, n_shadow=1):
    # For experiment on the class or feature balance
    # first split the dataset into two parts. 
    # class balance is measured on class=1.
    # feature balance is measured over first 5 features.
    # balance is measured on the group of records that has the feature value "<=.5" for each of the five features.
    if exp == 'class' or exp == 'feature':
        if exp == 'class':
            _size= int(cbal* trsize)
            #print(_size)
    
            dt0=data[data['class']==0].reset_index(drop=True)
            dt1=data[data['class']==1].reset_index(drop=True)
        
        if exp == 'feature':
            _size= int(fbal* trsize)
            #print(_size)
            
            dt1=data
            for i in range(0, 5):
                dt1=data[data['col'+str(i)]<=.5]
            temp_ind=np.setdiff1d(list(data.index), list(dt1.index))
            dt0= data.iloc[temp_ind,:].reset_index(drop=True)
            dt1=dt1.reset_index(drop=True)
    
        indices1=np.arange(len(dt1))
        indices0=np.arange(len(dt0))
            
        np.random.shuffle(indices0)        
        target_indices = random.sample(indices0.tolist(),trsize-_size)
        shadow_indices = np.setdiff1d(indices0, target_indices)    
        target_d0=dt0.iloc[target_indices,:].reset_index(drop=True)
        shadow_d0=dt0.iloc[shadow_indices,:].reset_index(drop=True)
        
              
        np.random.shuffle(indices1)
        target_indices = random.sample(indices1.tolist(),_size)
        shadow_indices = np.setdiff1d(indices1, target_indices)
        target_d1=dt1.iloc[target_indices,:].reset_index(drop=True)
        shadow_d1=dt1.iloc[shadow_indices,:].reset_index(drop=True)
        
                    
        target_data=pd.concat([target_d0,target_d1]).reset_index(drop=True)       
        tgt_ind=list(target_data.index)
        np.random.shuffle(tgt_ind)
        target_data = target_data.iloc[tgt_ind, :].reset_index(drop=True)
        
        shadow_data=pd.concat([shadow_d0,shadow_d1]).reset_index(drop=True)
        shd_ind=list(shadow_data.index)
        np.random.shuffle(shd_ind)
        shadow_data = shadow_data.iloc[shd_ind, :].reset_index(drop=True)       
    else:
        if exp = 'feat_no' or exp = 'entropy':
            data = data.iloc[:, 0:feat_no]
        
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
        
def exp_datasize(data, minsize, maxsize, step):
    for trsize in range (minsize, maxsize+1, step):
        shsize=minsize                
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='datasize')
        entropy, class_bal, class_indices, feat_bal, feat_indices = get_deafults(target_data)
        target, atk = mia_exec()
        return trsize, shsize, defaults, target, atk
    
    for shsize in range (minsize, maxsize+1, step):
        trsize=minsize
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='datasize')
        defaults= get_deafults(target_data)
        target, atk = mia_exec()
        return trsize, shsize, defaults, target, atk

def exp_classbalance(data, datasize):
    trsize = shsize = datasize
    
    for cbal in range (.1, 1, .1):
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='class', cbal = cbal)               
        defaults = get_deafults(target_data)
        target, atk = mia_exec()
        return trsize, shsize, defaults, target, atk

def exp_featurebalance(data, datasize):
    trsize = shsize = datasize
    
    for fbal in range (.1, 1, .1):
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='feature', fbal = fbal)           
        defaults = get_deafults(target_data)
        target, atk = mia_exec()
        return trsize, shsize, defaults, target, atk
        
def exp_featno(data, datasize, toal_feat):
    trsize = shsize = datasize
    
    for feat_no in range (1,toal_feat+1, 1):
        target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data, exptype='feat_no', feat_no = feat_no)
        defaults = get_deafults(target_data)
        target, atk = mia_exec()
        return trsize, shsize, feat_no, defaults, target, atk

def exp_entropy(data, datasize, toal_feat):
    return exp_featno(data, datasize, toal_feat)



exp = ['arch', 'combination', 'mutual_info', 'mia_ind', 'fairness']
        
def main(datalabel, exp_type):
    filename='C:\\Users\\45054541\\projects\\Cleans\\'+datalabel
    savefile=datalabel+'.csv'
    dt=pd.read_csv(filename+'.csv')
    
    for itr in range(0,100):
        if exp_type == 'datasize':
            minsize =1000
            maxsize = 10000
            step=1000
            trsize, shsize, defaults, target, atk = exp_datasize(data, minsize, maxsize, step)
        
        elif exp_type == 'class':
            datasize = 10000
            trsize, shsize, defaults, target, atk = exp_classbalance(data, datasize)
            
        elif exp_type == 'feature':
            datasize = 10000
            trsize, shsize, defaults, target, atk = exp_featurebalance(data, datasize)
            
        elif exp_type == 'feat_no':
            datasize = 10000
            trsize, shsize, feat_no, defaults, target, atk = exp_featurebalance(data, datasize)
            
        elif exp_type == 'entropy':
            datasize = 10000
            trsize, shsize, feat_no, defaults, target, atk = exp_entropy(data, datasize)
                
        fpr,tpr,roc_auc,ts_prec,ts_rec, ts_fbeta, tr_acc, ts_acc=target
        atk_prec, atk_rec, atk_acc, class_acc=atk#, c_atk_acc=result        
            
    result = trsize, shsize, feat_no    
    
    #st+='avg_mi'+str(avg_mi)+',max_mi'+str(max_mi)+',g'+str(group)+',p'+str(pred)+',i'+str(ind)+',mem'+str(indist)+','
    st+='g'+str(group)+',p'+str(pred)+',i'+str(ind)+',mem'+str(indist)+','
    st+='prec'+str(ts_prec)+',rec'+str(ts_rec)+',train_acc'+str(tr_acc)+',test_acc'+str(ts_acc)+',auc'+str(roc_auc)+','
    #print('auc:', roc_auc)
#                 for j in range(0, len(fpr)):
#                     st1+='fpr'+str(fpr[j])+','
#                 for j in range(0, len(tpr)):
#                     st1+='tpr'+str(tpr[j])+','
                

    st+='atk_prec'+str(atk_prec)+',atk_rec'+str(atk_rec)+',atk_acc'+str(atk_acc)+',c1_atk_acc'+ str(class_acc[0])+',c0_atk_acc'+ str(class_acc[0])
    #+',f1_atk_acc'+ str(feat_acc[0])+',f0_atk_acc'+ str(feat_acc[1])

    text_file=open('feat_'+savefile, 'a')
    text_file.write(st)
    text_file.write('\n')
    text_file.close()
    #print('time:', _time)
    
    

if __name__ == '__main__':

    main("")