

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    #if shuffle:
        #indices = np.arange(len(inputs))
        #np.random.shuffle(indices)
    temp=int(len(inputs)/batch_size)
    last_batch=len(inputs) % batch_size
    for i in range(0, len(inputs) - batch_size + 1, batch_size):
        if temp==1 and last_batch>0: excerpt = slice(i, i + batch_size + last_batch)
        #if shuffle: excerpt = indices[i:i + batch_size]
        else: excerpt = slice(i, i + batch_size)
        temp-=1
        yield inputs[excerpt], targets[excerpt].astype(np.int32)

def get_nn_model(input_var, n_in, n_hidden, n_out, act_func):
    net = dict()
    net['input'] = lasagne.layers.InputLayer( shape=(None,n_in), input_var=input_var)
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=act_func)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def train(dataset, batch_size, n_hidden, epochs, learning_rate, l2_ratio, activation='tanh',rtn_layer=True, attack=False, target=False, save_params=None):
    train_x= dataset[0]
    train_y= dataset[1]
    test_x= dataset[2]
    test_y= dataset[3]

    n_in = train_x.shape[1] #no of features
    n_out = len(np.unique(train_y)) #number of class
    

    if batch_size > len(train_y): batch_size = len(train_y)

    input_var = T.matrix('x')
    target_var = T.ivector('y')
    l_rate=T.scalar('l_rate')
    
    if activation=='ReLU': act_func=lasagne.nonlinearities.rectify
    else: act_func=lasagne.nonlinearities.tanh
    
    net = get_nn_model(input_var, n_in, n_hidden, n_out, act_func)
    net['input'].input_var=input_var 
    output_layer=net['output']
    
    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var).mean()
    loss = loss + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,lasagne.regularization.l2)
    
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, l_rate)
    train_fn = theano.function([input_var, target_var, l_rate], loss, updates=updates)
    
    # use trained network for predictions
    get_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    pred_fn = theano.function([input_var], get_prediction)
    
    #print ('Training...') 
    lr_decay = .1
    lr=learning_rate
    for epoch in range(epochs):        
        loss = 0
        train_pred_y = []
        #if attack==True and epoch!=0: lr=learning_rate * (lr_decay ** epoch)
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss+=train_fn(input_batch, target_batch, lr)
            pred = pred_fn(input_batch)
            train_pred_y.append(np.argmax(pred, axis=1))
        train_pred_y = np.concatenate(train_pred_y)
#     if(target):
#         #I(x;Theta)
#         st, savefile=save_params
#         par=params[0].get_value()      
#         coefs=[]
#         mi=[]
#         for i in range(0,len(par)):
#             p=max(par[i])#par[i].sum()/len(par[i])
#             temp=par[i] 
#             st1=st+'f'+str(i)
#             for j in range(0, n_hidden):
#                 st1+=','+str(temp[j])
#             st1+=',avg_'+str(par[i].sum()/len(par[i]))
#             st1+=',max_'+str(p)+'\n'
#             coefs.append(p)
#             text_file=open('params_'+savefile, 'a')
#             text_file.write(st1)
#             text_file.close()
        #print(st)
        #sys.exit()
#         coefs=np.array(coefs)
#         train_x=train_x.reset_index(drop=True)
#         #temp_ind= random.sample(list(np.arange(len(train_x))), 10000)
#         #temp_x=train_x.iloc[temp_ind,:].reset_index(drop=True)
#         for j in range(0,len(train_x)):
#             r=np.array(train_x.iloc[j,:]).reshape(-1,1)
#             _mi=mutual_info_regression(r, coefs)#, discrete_features=True)
#             mi.append(_mi[0])
#         avg_mi=sum(mi)/len(mi)
#         max_mi=max(mi)
#         print('avg_mi', avg_mi)
#         print('max_mi', max_mi)
    avg_mi=0
    max_mi=0
    #print ('Testing...')
    test_pred_y = []
    if batch_size > len(test_y): batch_size = len(test_y)

    for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
        pred = pred_fn(input_batch)
        test_pred_y.append(np.argmax(pred, axis=1))
    test_pred_y = np.concatenate(test_pred_y)
    
    
    if rtn_layer:
        if target: data= output_layer, train_pred_y, test_pred_y, avg_mi, max_mi
        else: data= output_layer, train_pred_y, test_pred_y
    else:
        data= train_pred_y, test_pred_y
    
    return data


#-----------------Dataset Generation-------------        
        
def target_shadow_split(feat_len,n_shadow):
    
    target_size = int(feat_len/3)

    indices = np.arange(feat_len).tolist()
    target_data_indices = random.sample(indices, target_size)
    shadow_data_indices = np.setdiff1d(indices, target_data_indices)
    shadow_all_indices=[]
    for i in range(0,n_shadow): shadow_all_indices.append(random.sample(shadow_data_indices.tolist(), target_size*2)) #target_size))
    indices= target_data_indices, shadow_all_indices
    
    return indices

#-------------------Evaluation--------------------
def TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))
def FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))
def FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))
def TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))