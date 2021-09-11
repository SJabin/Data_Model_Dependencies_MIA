import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import math

def get_fairness(X,y,pred_y, test_x=None, test_y=None, test_pred_y=None):
    if test_x is not None:
        X=pd.concat([X, test_x])
        y=np.concatenate((y,test_y), axis=0)
        pred_y=np.concatenate((pred_y,test_pred_y), axis=0)
    X=X.reset_index(drop=True)
    #y=y.reset_index(drop=True)
    
    #if len(feat_no)>4: feats = random.sample(feat_no, 4)
    #else: feats= feat_no
    
    temp=X
    #for i in feats:
    for i in range(0, 5):#(0,len(feats)):
        temp=temp[temp.iloc[:, i]<=.5]#.reset_index(drop=True)
        
    gid=list(temp.index)
    _gid=np.setdiff1d(np.arange(len(X)), gid)
        
#     #group fairness
#     Group=[]
#     if len(gid)!=0: 
#         g_pred_y=pred_y[gid]
#         _, pred_counts=np.unique(g_pred_y, return_counts=True)

#         for i in range(0, len(pred_counts)):
#             if pred_counts[i]!=0:
#                 Group.append(pred_counts[i]/len(g_pred_y))
#             else:
#                 Group.append(0)
    
#     _Group=[]
#     if len(_gid)!=0: 
#         _g_pred_y=pred_y[_gid]
#         _, _pred_counts=np.unique(_g_pred_y, return_counts=True)

#         for i in range(0, len(_pred_counts)):
#             if _pred_counts[i]!=0:
#                 _Group.append(_pred_counts[i]/len(_g_pred_y))
#             else:
#                 _Group.append(0)
    
#     g=0
#     for i in range (0, len(Group)):
#         if len(_Group)>i:
#             g+=abs(Group[i] - _Group[i])
#         else:
#             g+=abs(Group[i])
#     g_fair= g/2
    
#     #predictive fairness
#     #print('y[gid]',len(y[gid]))
#     #print('pred_y[gid]',len(pred_y[gid]))
    
#     #print('y[_gid]',len(y[_gid]))
#     #print('pred_y[_gid]',len(pred_y[_gid]))
    
#     if len(gid)!=0:
#         G_prec, _,_,_=precision_recall_fscore_support(y[gid],pred_y[gid], average='binary')
#     else:
#         G_prec=0
#     if len(_gid)!=0:
#         _G_prec, _,_,_=precision_recall_fscore_support(y[_gid],pred_y[_gid], average='binary')
#     else:
#         _G_prec=0
    
#     p_fair=abs(G_prec-_G_prec)

#     #individual fairness
#     Group_prob=len(gid)/len(X)
#     _Group_prob=len(_gid)/len(X)
#     x_distance=abs(Group_prob-_Group_prob) 
    
#     i_fair= abs(x_distance - g_fair)

#     return g_fair,p_fair,i_fair

#     gid=[]
#     _gid=[]
#     for x in range(0,len(X)):
#         if train_x[x,feat_no]==1:  
#             gid.append(x)
#         else: 
#             _gid.append(x)
    ix_prob=len(gid)/len(X)
    _ix_prob=len(_gid)/len(X)
    x_diff=abs(ix_prob-_ix_prob)            
    
    
    #group fairness
    count=0
    count0=0 #for individual fairness
    if len(gid)!=0: 
        g_y=y[gid]
        g_pred_y=pred_y[gid]
        for i in range(0,len(g_y)):
            if g_pred_y[i]==1: count+=1
            else: count0+=1
       
    _count=0
    _count0=0
    if len(_gid)!=0: 
        _g_y=y[_gid]
        _g_pred_y=pred_y[_gid]
        for i in range(0,len(_g_y)):
            if _g_pred_y[i]==1: _count+=1
            else: _count0+=1
  
    #predictive fairness    
    g_prec, g_rec, _, _=precision_recall_fscore_support(y[gid],pred_y[gid], average='binary')
    _g_prec, _g_rec, _, _=precision_recall_fscore_support(y[_gid],pred_y[_gid], average='binary')
    
#     test_gid=[]
#     _test_gid=[]
#     for x in range(0,len(test_x)):
#         if test_x[x,feat_no]==1:  
#             test_gid.append(x)
#         else: 
#             _test_gid.append(x)
    
#     #group fairness
#     if len(test_gid)!=0:
#         g_y=test_y[test_gid]
#         g_pred_y=test_pred_y[test_gid]
#         for i in range(0,len(g_y)):
#             if g_pred_y[i]==1: count+=1
#             else: count0+=1
        

#     if len(_test_gid)!=0:
#         _g_y=test_y[_test_gid]
#         _g_pred_y=test_pred_y[_test_gid]
#         for i in range(0,len(_g_y)):
#             if _g_pred_y[i]==1: _count+=1
#             else: _count0+=1
        
    if(count==0):
        g_prob=0
    else:
        g_prob=count/len(gid)
    
    if(count0==0):
        g_prob0=0
    else:
        g_prob0=count0/len(gid)
    
    if(_count==0): 
        _g_prob=0
    else:
        _g_prob=count/len(_gid)
    
    if(_count0==0): 
        _g_prob0=0
    else:
        _g_prob0=count0/len(_gid)
    
    iy_prob=max(g_prob,g_prob0)
    _iy_prob=max(_g_prob,_g_prob0)    
    y_diff=abs(iy_prob-_iy_prob)
    
#     _prec, _rec=precision_recall_fscore(test_y[test_gid],test_pred_y[test_gid])
#     g_prec+=_prec
#     g_prec/=2
    
#     _prec, _rec=precision_recall_fscore(test_y[_test_gid],test_pred_y[_test_gid])
#     _g_prec+=_prec
#     g_prec/=2
                       
#     st+=',x_'+str(x_diff)+',y_'+str(y_diff)+','+str()
#     st+=',g_'+str(g_prob)+',_g_'+str(_g_prob)+','+str()
#     st+=',g_prec_'+str(g_prec)+',_g_prec_'+str(_g_prec)+','+str()

    g_fair=abs(g_prob-_g_prob)
    p_fair=abs(g_prec-_g_prec)
    i_fair=abs(x_diff-y_diff)
    
    return g_fair,p_fair,i_fair


def get_membership_indistinguishability(pred_y, test_pred_y):
    labels, member_count=np.unique(pred_y, return_counts=True)
    _labels, nonmember_count=np.unique(test_pred_y, return_counts=True)
    
    member_probs=member_count / len(pred_y)
    nonmember_probs=nonmember_count / len(test_pred_y)
    prob_rate=[]
    
    prob_rate.append(abs(math.log(member_probs[0]/nonmember_probs[0])))
    prob_rate.append(abs(math.log(member_probs[1]/nonmember_probs[1])) if (len(member_probs)>1 and len(nonmember_probs)>1) else 0)
    
    #for i in range(0,len(member_probs)):
        #prob_rate.append(abs(math.log(member_probs[i]/nonmember_probs[i])) if (i>len(nonmember_probs)-1 or nonmember_probs[i]!=0) else 0)
    indistinguishability= max(prob_rate)
    
    return indistinguishability
   