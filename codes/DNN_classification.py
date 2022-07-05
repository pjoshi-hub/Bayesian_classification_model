

import tensorflow as tf   
import random
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

"""# DNN"""

def placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,shape=[n_x,None],name='X')
    Y = tf.placeholder(tf.float32,shape=[n_y,None],name='Y')
    return (X,Y)

def initialization(layers_dims,regularizer):
    parameters={}
    for l in range(len(layers_dims)-1):
        parameters["W"+str(l+1)] = tf.get_variable('W'+str(l+1),[layers_dims[l+1],layers_dims[l]],initializer = tf.contrib.layers.xavier_initializer(),regularizer=regularizer )
        parameters["b"+str(l+1)] = tf.get_variable("b"+str(l+1),[layers_dims[l+1],1],initializer = tf.zeros_initializer())
    return parameters

def forward_propagation(X,layers_dims,parameters):
    store = {}
    store["A0"] = X
    for l in range(len(layers_dims)-1):
        store["Z"+str(l+1)] = tf.add(tf.matmul(parameters["W"+str(l+1)],store["A"+str(l)]),parameters["b"+str(l+1)])
        store["A"+str(l+1)] = tf.nn.sigmoid(store["Z"+str(l+1)])

        if (l == len(layers_dims) - 2):
            return store["Z"+str(l+1)]
        else:
            store["A"+str(l+1)] = tf.nn.sigmoid(store["Z"+str(l+1)])

def compute_cost(ZL,Y,regularizer):
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.transpose(ZL),labels = tf.transpose(Y)))
    cost  = cost + reg_term
    return cost

def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,print_cost,layers_dims):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    n_x = X_train.shape[0]
    m = X_train.shape[1]
    n_y = Y_train.shape[0]
    costs =[]
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    (X,Y) = placeholders(n_x,n_y)
    parameters = initialization(layers_dims,regularizer)
    ZL = forward_propagation(X,layers_dims,parameters)
    cost = compute_cost(ZL,Y,regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)   
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            _,epoch_cost = sess.run([optimizer,cost],feed_dict={X : X_train, Y : Y_train})
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        logit_final_temp = tf.nn.softmax(ZL,axis=0)
        logit_final_test = logit_final_temp.eval(feed_dict={X: X_test})
        logit_final_train = logit_final_temp.eval(feed_dict = {X: X_train})
        arr_pred_train = np.argmax(logit_final_train, axis= 0)
        arr_pred_test = np.argmax(logit_final_test,axis=0)
        train_accuracy =  np.sum(arr_pred_train == Ytrain) / len(arr_pred_train)
        test_accuracy = np.sum(arr_pred_test == Ytest) / len(arr_pred_test)

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
    return [train_accuracy,test_accuracy,logit_final_test,logit_final_train,arr_pred_test,arr_pred_train]

def split_train_test(df_pca2,rs):
    for l in range(31):
        df_temp = df_pca2.loc[df_pca2['label'] == l]
        if l == 0:
            X_train, X_test, y_train, y_test = train_test_split(df_temp.iloc[:,:-1],df_temp.iloc[:,-1], test_size=0.20, random_state=rs)
        else:
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(df_temp.iloc[:,:-1],df_temp.iloc[:,-1], test_size=0.20, random_state=rs)
            X_train = pd.concat([X_train,X_train_temp])
            X_test = pd.concat([X_test,X_test_temp])
            y_train = pd.concat([y_train,y_train_temp])
            y_test = pd.concat([y_test,y_test_temp])
    return (X_train, X_test, y_train, y_test)

df_pca2 = pd.read_csv('100_genes_pca2.csv')

"""# Test"""

np.random.seed(0)
r = np.random.randint(500)
(X_train, X_test, y_train, y_test) = split_train_test(df_pca2,r)
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,55,31]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.001,num_epochs=1500,print_cost=True,layers_dims = layers_dims)

"""# External Validation"""

np.random.seed(0)
r = np.random.randint(500)
(X_train, X_test, y_train, y_test) = split_train_test(df_pca2,r)
df_test = pd.read_csv('combined_external.csv')
df_test = df_test[df_test['label'] == 4] 
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,55,31]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.001,num_epochs=1500,print_cost=True,layers_dims = layers_dims)

"""# Logistic Regression"""

np.random.seed(0)
r = np.random.randint(500)
(X_train, X_test, y_train, y_test) = split_train_test(df_pca2,r)
clf_log = LogisticRegression(penalty='l2').fit(X_train,y_train)
clf_log.score(X_test,y_test)
