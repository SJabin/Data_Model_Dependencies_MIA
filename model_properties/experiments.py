#from classifier import train as train_model, iterate_minibatches

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

    
def mia_exec(target_data, shadow_data, shadow_all_indices, layer=1, node=None, lrate=None, l2ratio=None, model = 'ANN', save_params = None):
    #train target
    
    # model architecture experiments 
    if node != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, n_hidden = node)
    
    if lrate != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, lrate = lrate)
    
    if l2ratio != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, l2ratio = l2ratio)
    
    
    # target-attack model combinations
    if model == 'ALL':
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data)#, n_layer=layer, l2ratio = l2ratio)
    else:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, model=model)

    #train shadow
    shadow_dataset=shadow_data, shadow_all_indices
    attack_train_x, attack_train_y, train_classes = train_shadow_model(shadow_dataset)
            
    #train attack
    attack_datasets=attack_train_x, attack_train_y, attack_test_x, attack_test_y
    classes=train_classes, test_classes                
    atk_results=train_attack(attack_datasets,classes)
    
    return target_results, atk_results        

    

def get_indices(trsize, shsize, data, exp=None, n_shadow=1):
    if exp == 'fairness':
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




        
def exp_arch(data, trsize, shsize, layer, node=None, lrate=None, l2ratio=None):    
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, layer = layer, node = node, lrate = lrate, l2ratio = l2ratio)
    return trsize, shsize, defaults, target, atk
    
    
def exp_combination(data, trsize, shsize, layer, model):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, model = model)
    return trsize, shsize, defaults, target, atk

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


def save_results(itr, result, savefile):
    trsize, shsize, defaults, target, atk = result
    
    fpr,tpr,roc_auc,ts_prec,ts_rec, ts_fbeta, tr_acc, ts_acc=target
    atk_prec, atk_rec, atk_acc, class_acc=atk#, c_atk_acc=result
    
    result = str(itr)+','+str(trsize)+','+str(shsize)+','#+ str(feat_no)+ ','
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
    
    
    
    

#exp = ['n_nodes','l_rates', 'l2_ratios', 'combination', 'mutual_info', 'mia_ind', 'fairness']
# Experiments on the n_nodes, l_rates, l2_ratios are done on ANNs having 1 to 5 hidden layers.

def main(datalabel, exp):
    filename='./'+datalabel+'.csv'
    savefile=exp+'_'+datalabel+'.csv'
    data = pd.read_csv(filename) 
    trsize = shsize = 10000
    
    for itr in range(0,2):
        print("\nIteration: ", itr)
        
        if exp == 'n_nodes':            
            print("\nExp: number of nodes============")
            nodes = [ 5, 50, 100, 500, 1000 ]
            
            # NN -> 1 to 5 hidden layers
            for layer in range(1,6):
                print("\nhidden layers: ", layer)
                for node in nodes: 
                    print("\nnodes per layer: ", node)
                    result = exp_arch(data, trsize, shsize, layer, node=node)
                    save_results(itr, result, savefile)
                
        if exp == 'l_rates':
            print("\nExp: learning rates============")
            lrates = [ .00001, .0001, .001, .01, .1 ]
            
            # NN -> 1 to 5 hidden layers
            for layer in range(1,6):
                print("\nhidden layers: ", layer)
                for lrate in lrates: 
                    print("\nlearning rate: ", lrate)
                    result = exp_arch(data, trsize, shsize, layer, lrate=lrate)
                    save_results(itr, result, savefile)
            
        if exp == 'l2_ratios':
            print("\nExp: L2 ratios============")
            l2ratios = [ 0, .001, .01, .1, 1, 2, 3 ]
            
            # NN -> 1 to 5 hidden layers
            for layer in range(1,6):
                print("\nhidden layers: ", layer)
                for l2ratio in l2ratios: 
                    print("\nl2 ratio: ", l2ratio)
                    result = exp_arch(data, trsize, shsize, layer, l2ratio=l2ratio)
                    save_results(itr, result, savefile)

        
        elif exp == 'combination':
            print("Exp: Model-model Combinations============")
            models = ['ANN', 'LR', 'SVC', 'RF', 'KNN', 'ALL']
            
            for model in models:
                print("\nl2 ratio: ", l2ratio)
                    result = exp_combination(data, trsize, shsize, layer, model=model)
                    save_results(itr, result, savefile)
            

        elif exp == 'mutual_info':
            print("Exp: Mutual Information============")

            
        elif exp == 'mia_ind':
            print("Exp: MIA Indistinguishability============")
            
        elif exp == 'fairness':
            print("Exp: Fairness Difference============")
        
               
            
    
   
    

if __name__ == '__main__':

    main("")