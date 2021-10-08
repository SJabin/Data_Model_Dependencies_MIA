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


# Membership Inference Attack 
# executed in three steps:
# training the target model -> train_target
# training the shadow model -> train_shadow, and,
# run attack -> train_attack 
# attack accuracy is the test_accuracy of the attack model. 

def mia_exec(target_data, shadow_data, shadow_all_indices, layer=1, node=None, lrate=None, l2ratio=None, tgt_model='ANN', sh_model='ANN', arch = False, save_params = None, mia_ind = False, fairness = False ):
    #train target
    
    # model architecture experiments 
    if node != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, n_hidden=node, arch = arch)
    
    elif lrate != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, lrate=lrate, arch = arch)
    
    elif l2ratio != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, n_layer=layer, l2ratio=l2ratio, arch = arch)

        
    # mutual information between records and model parameters experiment
    elif save_params != None:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, model=tgt_model, save_params=save_params)
    
    
    # mia_indistingushability experimnet
    elif mia_ind:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, model=tgt_model, mia_ind=mia_ind)
    
    # mia_indistingushability experiment
    elif fairness:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, model=tgt_model, fairness=fairness)
    
    
    else:
        attack_test_x, attack_test_y, test_classes, target_results= train_target_model(target_data, model=tgt_model)

    #train shadow
    shadow_dataset=shadow_data, shadow_all_indices
    attack_train_x, attack_train_y, train_classes = train_shadow_model(shadow_dataset, model = sh_model)
            
    #train attack
    attack_datasets=attack_train_x, attack_train_y, attack_test_x, attack_test_y
    classes=train_classes, test_classes                
    atk_results=train_attack(attack_datasets,classes)
    
    return target_results, atk_results        

    

def get_indices(trsize, shsize, data, n_shadow=1):
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
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, arch= True, layer = layer, node = node, lrate = lrate, l2ratio = l2ratio)   
    return trsize, shsize, defaults, target, atk

def exp_combination(data, trsize, shsize, tgt_model='ANN', sh_model='ANN'):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, tgt_model=tgt_model, sh_model=sh_model)
    return trsize, shsize, defaults, target, atk

def exp_mutual_info(data, trsize, shsize, save_params):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, save_params=save_params)
    return trsize, shsize, defaults, target, atk

def exp_indistiguishability(data, trsize, shsize):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, mia_ind = True)    
    return trsize, shsize, defaults, target, atk

def exp_fairness(data, trsize, shsize):
    target_data, shadow_data, shadow_all_indices = get_indices(trsize, shsize, data)
    defaults = get_deafults(target_data)
    target, atk = mia_exec(target_data, shadow_data, shadow_all_indices, fairness = True)    
    return trsize, shsize, defaults, target, atk


def save_results(itr, result, savefile, exp, model = None):
    trsize, shsize, defaults, target, atk = result    
    atk_prec, atk_rec, atk_acc, class_acc= atk
    
    result_str = str(itr)+','
    
    if exp == 'n_nodes' or exp == 'l_rates' or exp == 'l2_ratios':
        fpr, tpr, roc_auc, ts_prec, ts_rec, ts_fbeta, tr_acc, ts_acc, arch = target
        layer, node, lrate, l2ratio = arch
        result_str += str(layer)+','+str(node)+','+str(lrate)+','+ str(l2ratio)+ ','
        
    if exp == 'model_combine':
        tgt_model, sh_model = model
        fpr, tpr, roc_auc, ts_prec, ts_rec, ts_fbeta, tr_acc, ts_acc = target
        result_str += str(tgt_model)+','+str(sh_model)+','
        
    if exp == 'mutual_info':
        fpr,tpr, roc_auc, ts_prec, ts_rec, ts_fbeta, tr_acc, ts_acc,  avg_mi, max_mi = target
        result_str += str(avg_mi)+','+str(max_mi)+','
        
    if exp == 'mia_ind':
        fpr,tpr, roc_auc, ts_prec, ts_rec, ts_fbeta, tr_acc, ts_acc,  indist = target
        result_str += str(indist)+','
        
    if exp == 'fairness':
        fpr,tpr, roc_auc, ts_prec, ts_rec, ts_fbeta, tr_acc, ts_acc,  g_fair, p_fair, i_fair = target
        result_str += str(g_fair)+','+str(p_fair)+','+str(i_fair)+','
    
    result_str += str(ts_prec)+','+str(ts_rec)+','+str(tr_acc)+','+str(ts_acc)+','+str(roc_auc)+','
    result_str += str(atk_prec)+','+str(atk_rec)+','+str(atk_acc)
    
    write(result_str, savefile)
    
    
    
def write(result, savefile):
    text_file=open(savefile, 'a')
    text_file.write(result)
    text_file.write('\n')
    text_file.close()
    


    

#exp = ['n_nodes','l_rates', 'l2_ratios', 'combination', 'mutual_info', 'mia_ind', 'fairness']

# Experiments on the n_nodes, l_rates, l2_ratios are performed on ANNs having 1 to 5 hidden layers.
n_layers = 5
nodes = [ 5, 50, 100, 500, 1000 ]
lrates = [ .00001, .0001, .001, .01, .1 ]
l2ratios = [ 0, .001, .01, .1, 1, 2, 3 ]     


# experiments on the different target and shadow model combinations
# shadow model 'ALL' refers to the classifier build by stacking all five models  
tgt_models = [ 'ANN', 'LR', 'SVC', 'RF', 'KNN']
sh_models = ['ANN', 'LR', 'SVC', 'RF', 'KNN','All']

# for all other experiments datasize is fixed. For Adult dataset, datasize =10000
# for Purchase and Texas datasets change the datasize to 100000
trsize = shsize = 10000


# path for the dataset
filepath = './'



def main(datalabel, exp):
    filename = filepath + datalabel + '.csv'
    savefile = exp + '_' + datalabel + '.csv'
    data = pd.read_csv(filename) 

    
    for itr in range(0,100):
        print("\nIteration: ", itr)
        
        if exp == 'n_nodes':            
            print("\nExp: number of nodes============")
            
            # NN -> 1 to 5 hidden layers
            for layer in range(1,n_layers+1):
                print("\nhidden layers: ", layer)
                for node in nodes: 
                    print("nodes per layer: ", node)
                    result = exp_arch(data, trsize, shsize, layer, node=node)
                    save_results(itr, result, savefile, exp)
                
        if exp == 'l_rates':
            print("\nExp: learning rates============")

            # NN -> 1 to 5 hidden layers
            for layer in range(1,n_layers+1):
                print("\nhidden layers: ", layer)
                for lrate in lrates: 
                    print("learning rate: ", lrate)
                    result = exp_arch(data, trsize, shsize, layer, lrate=lrate)
                    save_results(itr, result, savefile, exp)
            
        if exp == 'l2_ratios':
            print("\nExp: l2 ratios============")
                        
            # NN -> 1 to 5 hidden layers
            for layer in range(1,n_layers+1):
                print("\nhidden layers: ", layer)
                for l2ratio in l2ratios: 
                    print("l2 ratio: ", l2ratio)
                    result = exp_arch(data, trsize, shsize, layer, l2ratio=l2ratio)
                    save_results(itr, result, savefile, exp)

        
        if exp == 'model_combine':
            print("Exp: target - shadow model Combinations============")
            
            for tmodel in tgt_models:
                for smodel in sh_models:
                    print("\ntarget model: ", tmodel)
                    print("shadow model: ", smodel)
                    result = exp_combination(data, trsize, shsize, tgt_model=tmodel, sh_model=smodel)
                    models = tmodel, smodel
                    save_results(itr, result, savefile, exp, model = models)
            
        if exp == 'mutual_info':
            save_params = str(itr), savefile
            print("Exp: Mutual Information between the records and model parameters")
            result = exp_mutual_info(data, trsize, shsize, save_params)
            save_results(itr, result, savefile, exp)
            
        if exp == 'mia_ind':
            print("Exp: MIA Indistinguishability============")
            result = exp_indistiguishability(data, trsize, shsize)
            save_results(itr, result, savefile, exp)
            
        if exp == 'fairness':
            print("Exp: Fairness Difference============")
            result = exp_fairness(data, trsize, shsize)
            save_results(itr, result, savefile, exp)  
            
    
   
    

if __name__ == '__main__':

    main("")
