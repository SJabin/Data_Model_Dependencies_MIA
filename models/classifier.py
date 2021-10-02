import numpy as np
import pandas as pd
import theano.tensor as T
import lasagne
import theano
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator

from utils.model_utils import get_fairness, get_mia_indistinguishability


def train_ANN (dataset, epochs, batch_size, n_hidden, learning_rate, l2_ratio, n_layer = 1, activation='tanh',rtn_layer=True, save_params=None, mia_ind = False, fairness= False):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]
    
    model = ANN(epochs, batch_size, learning_rate, l2_ratio, n_hidden, n_layer)
    model = model.fit(train_x, train_y)
    train_pred_y = model.predict(train_x)
    test_pred_y = model.predict(test_x)
    

    if save_params != None:
        st, savefile = save_params
        par = model.params[0].get_value()
        #print('par:', par)
        
        coefs=[]
        mi=[]
        for i in range(0,len(par)):
            #p=max(par[i])#par[i].sum()/len(par[i])
            temp=par[i] 
            st1=st+',feat_'+str(i)
            for j in range(0, n_hidden):
                st1+=','+str(temp[j])
            st1+=',avg_'+str(par[i].sum()/len(par[i]))
            st1+=',max_'+str(max(par[i]))+'\n'
            coefs.append(max(par[i]))
                
            text_file=open('params_'+savefile, 'a')
            text_file.write(st1)
            text_file.close()
    
        coefs=np.array(coefs)
        train_x=train_x.reset_index(drop=True)
        
        for j in range(0,len(train_x)):
            r=np.array(train_x.iloc[j,:]).reshape(-1,1)
            _mi=mutual_info_regression(r, coefs)#, discrete_features=True)
            mi.append(_mi[0])
        avg_mi=sum(mi)/len(mi)
        max_mi=max(mi)
        print('avg_mi: ', avg_mi)
        print('max_mi: ', max_mi)
            
        result = avg_mi, max_mi
    
    if mia_ind:
        result=get_mia_indistinguishability(train_pred_y, test_pred_y)
        
    if fairness:
        group_diff,pred_diff,ind_diff=get_fairness(train_x,train_y, train_pred_y, test_x, test_y, test_pred_y)
        result = group_diff,pred_diff,ind_diff
    
    if rtn_layer:
        if save_params != None or mia_ind or fairness: 
            data= model.output_layer, train_pred_y, test_pred_y, result
        else:
            data= model.output_layer, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data


def train_LR(dataset, rtn_layer=True):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(train_x)  
    xtest = sc_x.transform(test_x) 
    
    logisticRegr = LogisticRegression()
    logisticRegr.fit(xtrain,train_y)

    train_pred_y = logisticRegr.predict(xtrain)
    test_pred_y = logisticRegr.predict(xtest)
    
    if rtn_layer:
        data= logisticRegr, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data

def train_SVC(dataset, rtn_layer=True):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(train_x)  
    xtest = sc_x.transform(test_x) 
    
    svclassifier = SVC(kernel='rbf', probability=True)
    svclassifier.fit(xtrain, train_y)
    
    train_pred_y = svclassifier.predict(xtrain)
    test_pred_y = svclassifier.predict(xtest)

    if rtn_layer:
        data= svclassifier, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data

def train_RF(dataset, rtn_layer=True):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(train_x)  
    xtest = sc_x.transform(test_x)    
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(xtrain, train_y)
    
    train_pred_y = clf.predict(xtrain)
    test_pred_y = clf.predict(xtest)
    
    if rtn_layer:
        data= clf, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data


def train_KNN(dataset, rtn_layer=True):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]
    
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(train_x)  
    xtest = sc_x.transform(test_x)    
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(xtrain, train_y)
    
    train_pred_y = classifier.predict(xtrain)
    test_pred_y = classifier.predict(xtest)
    
    if rtn_layer:
        data= classifier, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data


def iterate_minibatches(inputs, targets=None, batch_size=100):
    #assert len(inputs) == len(targets)
    
    temp=int(len(inputs)/batch_size)
    last_batch=len(inputs) % batch_size
    for i in range(0, len(inputs) - batch_size + 1, batch_size):
        if temp==1 and last_batch>0: excerpt = slice(i, i + batch_size + last_batch)
        else: excerpt = slice(i, i + batch_size)
        temp-=1
        
        if targets is None:
            yield inputs[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt].astype(np.int32)

        
        
def get_nn_model(input_var, n_in, n_hidden, n_out, act_func, n_layer=1):
    net = dict()
    net['input'] = lasagne.layers.InputLayer( shape=(None,n_in), input_var=input_var)
    
    #varying n_layers for model architecture experiments
    for i in range(0, n_layer):
        inp = net['input'] if i==0 else net['l'+str(i-1)]            
        net['l'+str(i)] = lasagne.layers.DenseLayer(inp,num_units=n_hidden,nonlinearity=act_func)
    
    net['output'] = lasagne.layers.DenseLayer(
        net['l'+str(i)],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

   

class ANN(ClassifierMixin, BaseEstimator):
    
    def __init__(self, epochs=10, batch_size=100, learning_rate=.001, l2_ratio=1e-7, n_hidden=50, n_layer=1):
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.l2_ratio=l2_ratio
        self.n_hidden=n_hidden
        self.n_layer=n_layer
        
    
    def fit(self, X, y):
        n_in = X.shape[1] #no of features
        n_out = len(np.unique(y)) #number of class
    

        if self.batch_size > len(y): self.batch_size = len(y)

        input_var = T.matrix('x')
        target_var = T.ivector('y')
        l_rate=T.scalar('l_rate')
    
        act_func=lasagne.nonlinearities.tanh
    
        net = get_nn_model(input_var, n_in, self.n_hidden, n_out, act_func, self.n_layer)
        net['input'].input_var=input_var 
        self.output_layer = net['output']
        self.input_var = input_var
    
        # create loss function
        prediction = lasagne.layers.get_output(self.output_layer)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
        loss = loss + self.l2_ratio * lasagne.regularization.regularize_network_params(self.output_layer,lasagne.regularization.l2)
    
        # create parameter update expressions
        self.params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        updates = lasagne.updates.sgd(loss, self.params, l_rate)
        train_fn = theano.function([input_var, target_var, l_rate], loss, updates=updates)
    
        #print ('Training...') 
        lr=self.learning_rate
        for epoch in range(self.epochs):        
            self.loss = 0
            train_pred_y = []
            for input_batch, target_batch in iterate_minibatches(X, y, batch_size = self.batch_size):
                self.loss+=train_fn(input_batch, target_batch, lr)
            
        return self
    
    def predict(self, X):
        pred_y = []
        input_var = T.matrix('x')
        if self.batch_size > len(X): self.batch_size = len(X)
        
        # use trained network for predictions
        get_prediction = lasagne.layers.get_output(self.output_layer, deterministic=True)
        pred_fn = theano.function([self.input_var], get_prediction)
        
        for input_batch in iterate_minibatches(X, batch_size = self.batch_size):
            pred = pred_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
    
        return pred_y


# For target model vs shadow models ; 
# Stacked shadow model -> 'All'
def train_stacked(dataset, rtn_layer = True):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]

    
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('svm', SVC(kernel='rbf', probability=True)))
    level0.append(('rf', RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)))
    level0.append(('knn', KNeighborsClassifier(n_neighbors=5)))
    level0.append(('ann', ANN()))

    # define meta learner model
    level1 = LogisticRegression()
    
    # define the stacking ensemble
    classifier = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, stack_method='predict')
    
    classifier.fit(train_x, train_y)
    
    train_pred_y = classifier.predict(train_x)   
    test_pred_y = classifier.predict(test_x)
    
    if rtn_layer:
        data= classifier, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data
    
            




