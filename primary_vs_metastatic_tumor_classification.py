

import pandas as pd
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

"""# BNN"""

def placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,shape=[n_x,None],name='X')
    Y = tf.placeholder(tf.float32,shape=[n_y,None],name='Y')
    return (X,Y)

def forward_propagation(X,layers_dims,param_normal):
    
    def sample_epsilons(param_normal):
        epsilons_W = []
        epsilons_b = []
        for i in range(len(layers_dims)-1):
            epsilons_W.append(tf.random_normal(shape=tf.shape(param_normal["mu_W"+str(i+1)]), mean=0., stddev=1.0))
            epsilons_b.append(tf.random_normal(shape=tf.shape(param_normal["mu_b"+str(i+1)]), mean=0., stddev=1.0))
        return epsilons_W,epsilons_b

    def transform_rhos(layers_dims,param_normal):
        for i in range(len(layers_dims)-1):
            param_normal["rho_W"+str(i+1)] = softplus(param_normal["rho_W"+str(i+1)])
            param_normal["rho_b"+str(i+1)] = softplus(param_normal["rho_b"+str(i+1)])
        return param_normal

    def make_gaussian_samples(param_normal,layers_dims,epsilons_W,epsilons_b):
        samples_W = []
        samples_b = []
        for i in range(len(layers_dims)-1):
            samples_W.append(tf.add(param_normal["mu_W"+str(i+1)],tf.multiply( param_normal["rho_W"+str(i+1)] , epsilons_W[i])))
            samples_b.append(tf.add(param_normal["mu_b"+str(i+1)] ,tf.multiply( param_normal["rho_b"+str(i+1)] , epsilons_b[i])))
        return samples_W, samples_b

    epsilons_W,epsilons_b = sample_epsilons(param_normal)
    param_normal = transform_rhos(layers_dims,param_normal)
    samples_W, samples_b =  make_gaussian_samples(param_normal,layers_dims,epsilons_W,epsilons_b)
    
    store = {}
    store['A0'] = X
    for l in range(len(layers_dims)-1):
        store["Z"+str(l+1)] = tf.add(tf.matmul(samples_W[l],store["A"+str(l)]),samples_b[l])
        if (l == len(layers_dims) - 2):
            return store["Z"+str(l+1)],samples_W,samples_b
        store["A"+str(l+1)] = tf.nn.sigmoid(store["Z"+str(l+1)])
        store["A"+str(l+1)] = tf.nn.dropout(store["A"+str(l+1)])

def initialization(layers_dims):
    param_normal = {}
    for l in range(len(layers_dims)-1):
        param_normal["mu_W"+str(l+1)] = tf.get_variable('mu_W'+str(l+1),[layers_dims[l+1],layers_dims[l]],initializer =  tf.random_normal_initializer(mean = 0.0,stddev = 0.1))
        param_normal["rho_W"+str(l+1)] = -15.5 + tf.get_variable("rho_W"+str(l+1),[layers_dims[l+1],layers_dims[l]],initializer = tf.zeros_initializer())
        param_normal["mu_b"+str(l+1)] = tf.get_variable('mu_b'+str(l+1),[layers_dims[l+1],1],initializer =  tf.random_normal_initializer(mean = 0.0,stddev = 0.1))
        param_normal["rho_b"+str(l+1)] =  -16.5 + tf.get_variable("rho_b"+str(l+1),[layers_dims[l+1],1],initializer = tf.zeros_initializer())
    return param_normal

def softplus(x):
    return tf.log(1.0 + tf.exp(x))

def log_gaussian(x, mu,sigma):
    return -0.5 * tf.log(2.0 * tf.constant(math.pi)) - tf.log(sigma) - tf.truediv(tf.multiply((x-mu),(x-mu)), (2.0 * tf.multiply(sigma,sigma)))

def prior(x):
    mean_prior = tf.constant(0.0)
    sigma_prior = tf.constant(1.0)
    return tf.reduce_sum(log_gaussian(x,mean_prior,sigma_prior))

def log_softmax_likelihood(ZL, y):
    return  -1 * tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y),logits=tf.transpose(ZL)))

def calculate_cost(layers_dims,samples_W,samples_b,param_normal,ZL,label_one_hot):
    log_likelihood_sum = log_softmax_likelihood(ZL, label_one_hot)
    log_prior_list = []
    log_var_posterior_list = []
    for i in range(len(layers_dims)-1):
        log_prior_list.append(prior(samples_W[i]))
        log_prior_list.append(prior(samples_b[i]))
        log_var_posterior_list.append(tf.reduce_sum(log_gaussian(samples_W[i],param_normal["mu_W"+str(i+1)],param_normal["rho_W"+str(i+1)])))
        log_var_posterior_list.append( tf.reduce_sum(log_gaussian(samples_b[i],param_normal["mu_b"+str(i+1)],param_normal["rho_b"+str(i+1)])))
    log_prior_sum = sum(log_prior_list)
    log_var_posterior_sum = sum(log_var_posterior_list)
    return 1/(X_train.shape[1]) * (log_var_posterior_sum - log_prior_sum -  log_likelihood_sum)

def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,print_cost,layers_dims):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    n_x = X_train.shape[0]
    m = X_train.shape[1]
    n_y = Y_train.shape[0]
    costs =[]
    (X,Y) = placeholders(n_x,n_y)
    param_normal = initialization(layers_dims)
    ZL,samples_W,samples_b = forward_propagation(X,layers_dims,param_normal)
    loss = calculate_cost(layers_dims,samples_W,samples_b,param_normal,ZL,Y_train)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss) 
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            _,epoch_cost = sess.run([optimizer,loss],feed_dict={X : X_train, Y : Y_train})
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        logit_final_temp = tf.nn.softmax(ZL,axis=0)
        list_logit_train = []
        list_logit_test = []
        for pred in range(0,500):
            logit_final_test = logit_final_temp.eval(feed_dict={X: X_test})
            logit_final_train = logit_final_temp.eval(feed_dict = {X: X_train})
            list_logit_train.append(logit_final_train)
            list_logit_test.append(logit_final_test)
            if pred == 0:
                arr_pred_train = np.argmax(logit_final_train, axis= 0)
                arr_pred_test = np.argmax(logit_final_test,axis=0)
                train_accuracy =  np.sum(arr_pred_train == Ytrain) / len(arr_pred_train)
                test_accuracy = np.sum(arr_pred_test == Ytest) / len(arr_pred_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    return [train_accuracy,test_accuracy,list_logit_test,list_logit_train,arr_pred_test,arr_pred_train]

def test_train_split_common(df_pca2,rs):
    df_pca2_1 = df_pca2[df_pca2['label'] == 1]
    df_pca2_0 = df_pca2[df_pca2['label'] == 0]
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_pca2_1.iloc[:,:-1],df_pca2_1.iloc[:,-1], test_size=0.20, random_state=rs)
    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(df_pca2_0.iloc[:,:-1],df_pca2_0.iloc[:,-1], test_size=0.20, random_state=rs)
    X_train_0 = X_train_0.reset_index(drop=True) 
    X_train = pd.concat([X_train_1,X_train_0]).reset_index(drop=True)
    X_test = pd.concat([X_test_1,X_test_0]).reset_index(drop=True)
    y_train = pd.concat([y_train_1,y_train_0]).reset_index(drop=True)
    y_test = pd.concat([y_test_1,y_test_0]).reset_index(drop=True)
    return X_train,X_test,y_train,y_test

def filter_1(logits,y_test,arr_pred_test,type_uncer,t,t_l):
    
        aleo_list = []
        for j in range(len(logits)):
            prob_list = []
            for i in range(logits[j].shape[1]):
                arg = np.argmax(logits[j][:,i])
                prob = logits[j][:,i][arg]
                prob_list.append(prob)
            aleo = list(np.array(prob_list) - np.square(np.array(prob_list)))
            aleo_list.append(aleo)
            
        epi_list = []
        for j in range(len(logits)):
            prob_list = []
            for i in range(logits[j].shape[1]):
                arg = np.argmax(logits[j][:,i])
                prob = logits[j][:,i][arg]
                prob_list.append(prob)
            epi_list.append(np.array(prob_list))
        epistemic_uncertainty = np.mean(np.square((np.array(epi_list)-np.mean(np.array(epi_list),axis=0))),axis=0)
        aleatoric_uncertainty = np.mean(np.array(aleo_list),axis=0)
        
        incorrect = []
        for i in range(len(list(arr_pred_test))):
            if (arr_pred_test[i] != np.array(y_test)[i]):
                incorrect.append(i)
                
        list_aleatoric_correct = []
        list_aleatoric_incorrect = []
        for i in range(len(aleatoric_uncertainty)):
            if (i in incorrect) == True:
                list_aleatoric_incorrect.append(aleatoric_uncertainty[i])
            if (i in incorrect) == False:
                list_aleatoric_correct.append(aleatoric_uncertainty[i])
                
        list_epistemic_correct = []
        list_epistemic_incorrect = []
        for i in range(len(epistemic_uncertainty)):
            if (i in incorrect) == True:
                list_epistemic_incorrect.append(epistemic_uncertainty[i])
            if (i in incorrect) == False:
                list_epistemic_correct.append(epistemic_uncertainty[i])
                
        corr = np.mean(np.array(list_aleatoric_correct))
        incorr = np.mean(np.array(list_aleatoric_incorrect))
        corr_epistemic = np.mean(np.array(list_epistemic_correct))
        incorr_epistemic = np.mean(np.array(list_epistemic_incorrect))
        
        if type_uncer == 'Aleatoric':
   
            return [aleatoric_uncertainty,corr,incorr]
         
        if type_uncer == 'Epistemic':
            
            return [epistemic_uncertainty,corr_epistemic,incorr_epistemic]

"""# Integrating and Preprocessing Phenotype Data"""

filelist = []
for file in glob.glob('Phenotype_data/*'):
    filelist.append(file)
l = []
for i in range(len(filelist)):
    l.append(filelist[i][31:])
df = pd.DataFrame()
df['filelist'] = l
df.to_csv('filelist.csv',index=False)
for i in range(len(filelist)):
    if i == 0:
        df = pd.read_csv('Phenotype_data/'+str(filelist[i][31:]),sep='\t')
    else:
        df_temp = pd.read_csv('Phenotype_data/'+str(filelist[i][31:]),sep='\t')
        df = pd.concat([df,df_temp],join='inner')
df_pca2 = pd.read_csv('103_genes_pca2.csv')
genes = list(df_pca2.columns[:-1])
df_samples = pd.read_csv('TCGA_samples.csv')
df_samples.columns = ['sampleID']
df_vital_status = df[['sampleID','_EVENT']]
df_sample_type = df[['sampleID','sample_type']]

df_sample_type_primary = df_sample_type[df_sample_type['sample_type'] == 'Primary Tumor']
df_sample_type_metastatic = df_sample_type[df_sample_type['sample_type'] == 'Metastatic']
df_sample_type_primary['sample_type'] = 0
df_sample_type_metastatic['sample_type'] = 1
df_phenotype_prediction = pd.concat([df_sample_type_primary,df_sample_type_metastatic])

df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common.drop('label',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['sample_type'])
df_pca2_common.drop('sample_type',axis=1,inplace=True)

np.random.seed(0)
r = np.random.randint(500)
X_train,X_test,y_train,y_test = test_train_split_common(df_pca2_common,r)
X_train['label'] = list(y_train)
c = 0
for i in range(len(y_test)):
    if y_test[i] == 0:
        c = c+1

"""# Primary vs Metastatic Prediction"""

X_train,X_test,y_train,y_test = test_train_split_common(df_pca2_common,r)
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,2]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)

"""# Minimising False Negatives"""

logits_T = np.transpose(logits[0])[:,1]
y_true = np.array(y_test)
precision_bayesian, recall_bayesian, thresholds_bayesian = precision_recall_curve(y_true, logits_T)
y_pred = []
for i in range(len(logits_T)):
    if logits_T[i] >= 0.019:
        y_pred.append(1)
    else:
        y_pred.append(0)
tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall)/(precision+recall)

"""# Maximising F1 Score"""

logits_T = np.transpose(logits[0])[:,1]
y_true = np.array(y_test)
precision_bayesian, recall_bayesian, thresholds_bayesian = precision_recall_curve(y_true, logits_T)
y_pred = []
for i in range(len(logits_T)):
    if logits_T[i] >= 0.161:
        y_pred.append(1)
    else:
        y_pred.append(0)
tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall)/(precision+recall)

"""# Uncertainty Analysis"""

[al_train,mu_train_ac,mu_train_nac] = filter_1(logits_train,y_train,arr_pred_train,'Aleatoric','Train',_)
[ep_train,mu_train_ec,mu_train_nec] = filter_1(logits_train,y_train,arr_pred_train,'Epistemic','Train',_)

[al_test,mu_test_ac,mu_test_nac] = filter_1(logits,y_true,y_pred,'Aleatoric','Test',mu_train_ac)
[ep_test,mu_test_ec,mu_test_nec] = filter_1(logits,y_true,y_pred,'Epistemic','Test',mu_train_ec)

train_index_1 = list(np.where(np.array(y_train)==1)[0])
train_index_0 = list(np.where(np.array(y_train)==0)[0])
al_mean_1 = np.mean(al_train[train_index_1])
al_mean_0 = np.mean(al_train[train_index_0])
ep_mean_1 = np.mean(ep_train[train_index_1])
ep_mean_0 = np.mean(ep_train[train_index_0])

"""# filter 1"""

list_0 = list(np.where(np.array(y_pred) == 0)[0])
al_test_0 = al_test[list_0]
ep_test_0 = ep_test[list_0]
list_pred_0 = np.array(y_pred)[list_0]
y_test_0 = np.array(y_true)[list_0]
corr = []
incorr = []
certain_pred_0 = []
for i in range(len(list_0)):
    if (list_pred_0[i] == y_test_0[i]):
        corr.append(i)
    else:
        incorr.append(i)
        
    if (al_test_0[i] <= np.mean(al_train)) and (ep_test_0[i] <= np.mean(ep_train)):
        certain_pred_0.append(y_test_0[i])

list_1 = list(np.where(np.array(y_pred) == 1)[0])
al_test_1 = al_test[list_1]
ep_test_1 = ep_test[list_1]
list_pred_1 = np.array(y_pred)[list_1]
y_test_1 = np.array(y_true)[list_1]
corr = []
incorr = []
certain_pred_1 = []
for i in range(len(list_1)):
    if (list_pred_1[i] == y_test_1[i]):
        corr.append(i)
    else:
        incorr.append(i)
    if (al_test_1[i] <= np.mean(al_train)) and (ep_test_1[i] <= np.mean(ep_train)):
        certain_pred_1.append(y_test_1[i])

"""# filter 2"""

list_0 = list(np.where(np.array(y_pred) == 0)[0])
al_test_0 = al_test[list_0]
ep_test_0 = ep_test[list_0]
list_pred_0 = np.array(y_pred)[list_0]
y_test_0 = np.array(y_true)[list_0]
corr = []
incorr = []
certain_pred_0 = []
for i in range(len(list_0)):
    if (list_pred_0[i] == y_test_0[i]):
        corr.append(i)
    else:
        incorr.append(i)
        
    if (al_test_0[i] <= al_mean_0) and (ep_test_0[i] <= ep_mean_0):
        certain_pred_0.append(y_test_0[i])

list_1 = list(np.where(np.array(y_pred) == 1)[0])
al_test_1 = al_test[list_1]
ep_test_1 = ep_test[list_1]
list_pred_1 = np.array(y_pred)[list_1]
y_test_1 = np.array(y_true)[list_1]
corr = []
incorr = []
certain_pred_1 = []
for i in range(len(list_1)):
    if (list_pred_1[i] == y_test_1[i]):
        corr.append(i)
    else:
        incorr.append(i)
    if (al_test_1[i] <= al_mean_1) and (ep_test_1[i] <= ep_mean_1):
        certain_pred_1.append(y_test_1[i])

"""# Number of samples plot"""

list_0 = list(np.where(np.array(y_pred) == 0)[0])
al_test_0 = al_test[list_0]
ep_test_0 = ep_test[list_0]
list_pred_0 = np.array(y_pred)[list_0]
y_test_0 = np.array(y_true)[list_0]
l_temp = np.arange(0.1,5.1,0.1)
fn_filter_1 = []
num_samples_filter_1 = []
for j in l_temp:
    corr = []
    incorr = []
    certain_pred_0 = []

    for i in range(len(list_0)):
        if (list_pred_0[i] == y_test_0[i]):
            corr.append(i)
        else:
            incorr.append(i)

        if (al_test_0[i] <= j*np.mean(al_train)) and (ep_test_0[i] <= j* np.mean(ep_train)):
            certain_pred_0.append(y_test_0[i])
    fn_filter_1.append(len(np.where(np.array(certain_pred_0)==1)[0]))
    num_samples_filter_1.append(len(certain_pred_0))

list_0 = list(np.where(np.array(y_pred) == 0)[0])
al_test_0 = al_test[list_0]
ep_test_0 = ep_test[list_0]
list_pred_0 = np.array(y_pred)[list_0]
y_test_0 = np.array(y_true)[list_0]
l_temp = np.arange(0.1,5.1,0.1)
fn_filter_2 = []
num_samples_filter_2 = []
for j in l_temp:
    corr = []
    incorr = []
    certain_pred_0 = []
    for i in range(len(list_0)):
        if (list_pred_0[i] == y_test_0[i]):
            corr.append(i)
        else:
            incorr.append(i)

        if (al_test_0[i] <= j*al_mean_0) and (ep_test_0[i] <= j* ep_mean_0):
            certain_pred_0.append(y_test_0[i])
    fn_filter_2.append(len(np.where(np.array(certain_pred_0)==1)[0]))
    num_samples_filter_2.append(len(certain_pred_0))

fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,fn_filter_1,c='r',label='False Negatives filter 1')
l2 = ax1.plot(l_temp,fn_filter_2,c='r',linestyle = '--',label='False Negatives filter 2')
ax1.tick_params(axis='y', labelcolor='r')
l3 = ax2.plot(l_temp,num_samples_filter_1,c='g',label = 'No. of samples filter 1' )
l4 = ax2.plot(l_temp,num_samples_filter_2,c='g',linestyle = '--',label='No. of samples filter 2')
ax2.tick_params(axis='y', labelcolor='g')
leg = l1+l2+l3+l4
ax1.set_ylabel('No. of false negatives', color='r',fontsize=15)
ax2.set_ylabel('No. of samples retained', color='g',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.45))
ax1.set_yticks(np.arange(0,5,1))
ax2.set_yticks(np.arange(0.1,1801,400))
ax2.grid()
ax1.grid()
#fig.savefig('paper_figures/'+str('times_train_uncertainty_plot_imbalanced_classification')+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()

"""# Precision-Recall curve"""

sns.set()
fig = plt.figure()
plt.plot(recall_bayesian,precision_bayesian)
plt.xlabel('Recall',fontsize=15)
plt.ylabel('Precision',fontsize=15)
#fig.savefig('paper_figures/precision_recall_curve.pdf',format='pdf',dpi=1200)
plt.show()