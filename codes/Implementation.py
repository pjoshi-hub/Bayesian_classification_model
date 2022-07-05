
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', './BNN.ipynb')
get_ipython().run_line_magic('run', './Uncertainty.ipynb')


# In[2]:


np.random.seed(0)
r = np.random.randint(500)


# In[3]:


import statsmodels.api as sm
from sklearn.metrics import precision_recall_fscore_support
sns.set_style('whitegrid')


# In[4]:


df_analysis = pd.read_csv('2.csv')


# In[5]:


sample_numbers = list(df_analysis.iloc[1:,1])


# In[6]:


analysis_type = 'all_cancer_types'
cancer_types = ['LAML','ACC','BLCA','LGG','BRCA','CESC','CHOL','COAD','UCEC','ESCA','GBM','HNSC','KIRC','KIRP','LIHC','LUAD','LUSC','DLBC','MESO','OV','PAAD','PCPG','PRAD','READ','SKCM','STAD','TGCT','THYM','THCA','UCS','UVM']


# In[ ]:


df_pca2 = pd.read_csv('100_genes_pca2_new.csv')
(X_train, X_test, y_train, y_test) = split_train_test(df_pca2,r)
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,31]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,grad_val] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[ ]:


dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-9,analysis_type)


# In[28]:


len(list_final_pred)/len(y_test)


# In[29]:


fscore_filter


# In[9]:


fig = plt.figure(figsize=(25,5))
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta')
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black')
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('Cancer types',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=12.5)
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
plt.show()


# In[10]:


sns.set_style('darkgrid')
fig = plt.figure(figsize=(25,5))
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.2,fscore,label = 'BNN',color='r',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.1,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.2,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('Cancer types',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=12.5)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
plt.show()


# In[11]:


ep_test = com_ep_un(logits)
acc_2 = []
l_temp = np.arange(0.1,20.1,0.1)
sam_2 = []
for j in l_temp:
    list_final_pred = []
    list_final_test = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]*1e9 <= j* dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])
    acc_2.append(accuracy_score(list_final_test,list_final_pred))
    sam_2.append(len(list_final_test)*100/2034)
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,np.array(acc_2),c='r',linestyle='-',label='Overall Accuracy')
ax1.tick_params(axis='y', labelcolor='r')
l2 = ax2.plot(l_temp,sam_2,c='b',linestyle = '-',label = 'No. of samples' )
ax2.tick_params(axis='y', labelcolor='b')
leg = l2+l1
ax1.set_ylabel('Overall Accuracy (%)', color='r',fontsize=15)
ax2.set_ylabel('% samples retained', color='b',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.77))
ax2.set_yticks(np.arange(0,110,10))
ax1.set_xticks(np.arange(0,21))
ax1.grid()
plt.show()


# # Lower Grade Glioma

# # Subtypes

# In[12]:


analysis_type = 'lgg_subtypes'
cancer_types = ['Oligodendroglioma','Astrocytoma','Oligoastrocytoma']


# In[13]:


df_pca2 = pd.read_csv('brain lower grade glioma',sep='\t')
df_samples = pd.DataFrame(list(df_pca2.columns)[1:])
df_samples.columns = ['sampleID']
df_pca2_new = df_pca2.iloc[:,1:].T.reset_index(drop=True)
genes = list(df_pca2['sample'])
df_pca2_new.columns = genes
df_pca2 = df_pca2_new
filelist = []
for file in glob.glob('Phenotypic_data1/*'):
    filelist.append(file)
    
for i in range(len(filelist)):
    if filelist[i][31:] =='brain lower grade glioma':
        print(i)
        break
df_temp = pd.read_csv('Phenotypic_data1/'+str(filelist[3][31:]),sep='\t')
df_temp = df_samples.merge(df_temp,on='sampleID',how='inner')
df_sample_type = df_temp[['sampleID','histological_type']]
df_sample_type_oligodendroglioma = df_sample_type[df_sample_type['histological_type'] == 'Oligodendroglioma']
df_sample_type_astrocytoma = df_sample_type[df_sample_type['histological_type'] == 'Astrocytoma']
df_sample_type_oligoastrocytoma = df_sample_type[df_sample_type['histological_type'] == 'Oligoastrocytoma']
df_sample_type_oligodendroglioma['histological_type'] = 0
df_sample_type_astrocytoma['histological_type'] = 1
df_sample_type_oligoastrocytoma['histological_type'] = 2
df_phenotype_prediction = pd.concat([df_sample_type_oligodendroglioma,df_sample_type_astrocytoma,df_sample_type_oligoastrocytoma])
df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['histological_type'])
df_pca2_common.drop('histological_type',axis=1,inplace=True)


# In[14]:


l = ['CCNL2',
 'BOK',
 'REST',
 'LOC100272228',
 'C11orf9',
 'OTUD3',
 'C1orf27',
 'NACC1',
 'CCNE2',
 'SLC2A1']   


# In[15]:


X_train,X_test,y_train,y_test = split_train_test(df_pca2_common,r)
X_train = X_train.reset_index(drop=True)
X_train=X_train[l]
X_test = X_test[l]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,Y_train.shape[0]]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,Z] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[92]:


sns.set_style('whitegrid')


# In[16]:


dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-10,'lgg_subtypes')


# In[17]:


sns.set_style('whitegrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta',width=0.2)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black',width=0.2)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('LGG subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1.45, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
plt.show()


# In[24]:


len(list_final_pred)/len(y_test)


# In[18]:


sns.set_style('darkgrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.2,fscore,label = 'BNN',color='r',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.1,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.2,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('LGG subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
plt.show()


# In[23]:


len(list_final_test)/len(y_test)


# In[20]:


plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='r')
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,bottom = fscore,label = 'BNN+Filter') 


# In[26]:


ep_test = com_ep_un(logits)
acc_2 = []
l_temp = np.arange(0.1,20.1,0.1)
sam_2 = []
for j in l_temp:
    list_final_pred = []
    list_final_test = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]*1e9 <= j* dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])
    acc_2.append(accuracy_score(list_final_test,list_final_pred))
    sam_2.append(len(list_final_test)*100/len(y_test))
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,np.array(acc_2),c='r',linestyle='-',label='Overall Accuracy')
ax1.tick_params(axis='y', labelcolor='r')
l2 = ax2.plot(l_temp,sam_2,c='b',linestyle = '-',label = 'No. of samples' )
ax2.tick_params(axis='y', labelcolor='b')
leg = l2+l1
ax1.set_ylabel('Overall Accuracy (%)', color='r',fontsize=15)
ax2.set_ylabel('% samples retained', color='b',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.77))
ax2.set_yticks(np.arange(0,110,10))
ax1.set_xticks(np.arange(0,21))
ax1.grid()
fig.savefig(('filtering_cutoff_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# # Breast Invasive Carcinoma

# In[46]:


analysis_type = 'brca_subtypes'
cancer_types = ['Infiltrating Ductal Carcinoma','Infiltrating Lobular Carcinoma']


# In[47]:


df_pca2 = pd.read_csv('breast invasive carcinoma',sep='\t')
df_samples = pd.DataFrame(list(df_pca2.columns)[1:])
df_samples.columns = ['sampleID']
df_pca2_new = df_pca2.iloc[:,1:].T.reset_index(drop=True)
genes = list(df_pca2['sample'])
df_pca2_new.columns = genes
df_pca2 = df_pca2_new
filelist = []
for file in glob.glob('Phenotypic_data1/*'):
    filelist.append(file)
c = 0    
for i in range(len(filelist)):
    if filelist[i][31:] =='breast invasive carcinoma':
        print(i)
        c=i
        break
df_temp = pd.read_csv('Phenotypic_data1/'+str(filelist[c][31:]),sep='\t')
df_temp = df_samples.merge(df_temp,on='sampleID',how='inner')
df_sample_type = df_temp[['sampleID','histological_type']]
df_sample_type_ductal = df_sample_type[df_sample_type['histological_type'] == 'Infiltrating Ductal Carcinoma']
df_sample_type_lobular = df_sample_type[df_sample_type['histological_type'] == 'Infiltrating Lobular Carcinoma']
df_sample_type_ductal['histological_type'] = 0
df_sample_type_lobular['histological_type'] = 1
df_phenotype_prediction = pd.concat([df_sample_type_ductal,df_sample_type_lobular])
df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['histological_type'])
df_pca2_common.drop('histological_type',axis=1,inplace=True)


# In[48]:


l = ['CD48',
 'IL2RG',
 'SELPLG',
 'ARHGAP15',
 'NCKAP1L',
 'CD2',
 'PYHIN1',
 'GZMA',
 'SAMD3',
 'PJA2',
 'C5orf41',
 'KIAA1109',
 'LSM4',
 'TTBK2',
 'TCP11L2',
 'FAM63B',
 'BOD1L',
 'PSMG3',
 'CLDN5',
 'KANK3',
 'USHBP1',
 'GRASP',
 'PLAC9',
 'AVPR2',
 'GRRP1',
 'GPIHBP1',
 'IGFBP6',
 'DEGS2',
 'MLPH',
 'PCSK4',
 'PRR15',
 'FBP1',
 'GATA3',
 'CCDC96',
 'P4HTM',
 'B3GNT5',
 'STAG3L3',
 'LENG8',
 'PVRIG',
 'RPL32P3',
 'AGAP6',
 'DMTF1',
 'KIAA0907',
 'COL1A2',
 'LOC100128842',
 'CDC42BPG',
 'PTK7',
 'PVRL4',
 'GRAMD2',
 'FBLIM1',
 'KRT7',
 'ZNF710',
 'SDC1',
 'CCNJL',
 'MUCL1',
 'SPINK8',
 'CEACAM7',
 'POF1B',
 'ALDH3B2',
 'ATP13A5',
 'ACPP',
 'ATP13A4',
 'CLCA2',
 'HCFC1',
 'FAM129B',
 'NACC2',
 'MITD1',
 'NCRNA00201',
 'PHF11',
 'TMEM184B',
 'MDM4',
 'WNT9A',
 'TTC22',
 'PDZK1IP1',
 'SS18L1',
 'GPR137C',
 'DNASE1',
 'NFIX',
 'C16orf87',
 'MSI1',
 'NKIRAS2',
 'CHMP4C',
 'COL2A1',
 'NAGS',
 'HAP1',
 'GABPB2',
 'ZBTB7B',
 'CDH1',
 'IDH1']


# In[49]:


X_train,X_test,y_train,y_test = split_train_test(df_pca2_common,r)
X_train = X_train.reset_index(drop=True)
X_train=X_train[l]
X_test = X_test[l]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,Y_train.shape[0]]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,Z] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[50]:


sns.set_style('whitegrid')


# In[51]:


dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-9,analysis_type)


# In[52]:


len(list_final_pred)/len(y_test)


# In[53]:


fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('BRCA subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig(str('baseline_filter_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[34]:


sns.set_style('darkgrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.1,fscore,label = 'BNN',color='r',width=0.05)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.05,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.1,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.05)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('BRCA subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig(str('method_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[35]:


ep_test = com_ep_un(logits)
acc_2 = []
l_temp = np.arange(0.1,20.1,0.1)
sam_2 = []
for j in l_temp:
    list_final_pred = []
    list_final_test = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]*1e9 <= j* dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])
    acc_2.append(accuracy_score(list_final_test,list_final_pred))
    sam_2.append(len(list_final_test)*100/len(y_test))
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,np.array(acc_2),c='r',linestyle='-',label='Overall Accuracy')
ax1.tick_params(axis='y', labelcolor='r')
l2 = ax2.plot(l_temp,sam_2,c='b',linestyle = '-',label = 'No. of samples' )
ax2.tick_params(axis='y', labelcolor='b')
leg = l2+l1
ax1.set_ylabel('Overall Accuracy (%)', color='r',fontsize=15)
ax2.set_ylabel('% samples retained', color='b',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.77))
ax2.set_yticks(np.arange(0,110,10))
ax1.set_xticks(np.arange(0,21))
ax1.grid()
fig.savefig(str('filtering_cutoff_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# # Corpus Endometroid Carcinoma

# In[54]:


analysis_type = 'coen_subtypes'
cancer_types =['Endometroid endometrial carcinoma','Serous endometrial carcinoma']


# In[55]:


df_pca2 = pd.read_csv('corpus endometrioid carcinoma',sep='\t')
df_samples = pd.DataFrame(list(df_pca2.columns)[1:])
df_samples.columns = ['sampleID']
df_pca2_new = df_pca2.iloc[:,1:].T.reset_index(drop=True)
genes = list(df_pca2['sample'])
df_pca2_new.columns = genes
df_pca2 = df_pca2_new
filelist = []
for file in glob.glob('Phenotypic_data1/*'):
    filelist.append(file)
c = 0    
for i in range(len(filelist)):
    if filelist[i][31:] =='corpus endometrioid carcinoma':
        print(i)
        c=i
        break
df_temp = pd.read_csv('Phenotypic_data1/'+str(filelist[c][31:]),sep='\t')
df_temp = df_samples.merge(df_temp,on='sampleID',how='inner')
df_sample_type = df_temp[['sampleID','histological_type']]
df_sample_type_endometroid = df_sample_type[df_sample_type['histological_type'] == 'Endometrioid endometrial adenocarcinoma']
df_sample_type_serous = df_sample_type[df_sample_type['histological_type'] == 'Serous endometrial adenocarcinoma']
df_sample_type_endometroid['histological_type'] = 0
df_sample_type_serous['histological_type'] = 1
df_phenotype_prediction = pd.concat([df_sample_type_endometroid,df_sample_type_serous])
df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['histological_type'])
df_pca2_common.drop('histological_type',axis=1,inplace=True)


# In[56]:


l = ['BUB1',
 'MELK',
 'MRPL52',
 'ZNF593',
 'FRMPD2',
 'DLEC1',
 'CARD11',
 'RNF213',
 'BIRC3',
 'APOL1',
 'MAST2',
 'HPCA',
 'FZD2',
 'AFG3L1',
 'MFI2',
 'RET',
 'ZCCHC18',
 'C4orf27',
 'SRCIN1',
 'FGF19']


# In[57]:


X_train,X_test,y_train,y_test = split_train_test(df_pca2_common,r)
X_train = X_train.reset_index(drop=True)
X_train=X_train[l]
X_test = X_test[l]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,Y_train.shape[0]]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,Z] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[58]:


sns.set_style('whitegrid')
dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-10,analysis_type)


# In[59]:


len(list_final_pred)/len(y_test)


# In[60]:


fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('COEN subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig('Desktop/paper_revision/'+str('baseline_filter_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[42]:


sns.set_style('darkgrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.1,fscore,label = 'BNN',color='r',width=0.05)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.05,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.1,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.05)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('COEN subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig('Desktop/paper_revision/'+str('method_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[43]:


ep_test = com_ep_un(logits)
acc_2 = []
l_temp = np.arange(0.1,20.1,0.1)
sam_2 = []
for j in l_temp:
    list_final_pred = []
    list_final_test = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]*1e9 <= j* dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])
    acc_2.append(accuracy_score(list_final_test,list_final_pred))
    sam_2.append(len(list_final_test)*100/len(y_test))
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,np.array(acc_2),c='r',linestyle='-',label='Overall Accuracy')
ax1.tick_params(axis='y', labelcolor='r')
l2 = ax2.plot(l_temp,sam_2,c='b',linestyle = '-',label = 'No. of samples' )
ax2.tick_params(axis='y', labelcolor='b')
leg = l2+l1
ax1.set_ylabel('Overall Accuracy (%)', color='r',fontsize=15)
ax2.set_ylabel('% samples retained', color='b',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.77))
ax2.set_yticks(np.arange(0,110,10))
ax1.set_xticks(np.arange(0,21))
ax1.grid()
fig.savefig(str('filtering_cutoff_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# # Esophageal Carcinoma

# In[61]:


analysis_type = 'esca_subtypes'
cancer_types =['Squamous Cell Carcinoma','Adenocarcinoma']


# In[62]:


df_pca2 = pd.read_csv('esophageal carcinoma',sep='\t')
df_samples = pd.DataFrame(list(df_pca2.columns)[1:])
df_samples.columns = ['sampleID']
df_pca2_new = df_pca2.iloc[:,1:].T.reset_index(drop=True)
genes = list(df_pca2['sample'])
df_pca2_new.columns = genes
df_pca2 = df_pca2_new
filelist = []
for file in glob.glob('Phenotypic_data1/*'):
    filelist.append(file)
c = 0    
for i in range(len(filelist)):
    if filelist[i][31:] =='esophageal carcinoma':
        print(i)
        c=i
        break
df_temp = pd.read_csv('Phenotypic_data1/'+str(filelist[c][31:]),sep='\t')
df_temp = df_samples.merge(df_temp,on='sampleID',how='inner')
df_sample_type = df_temp[['sampleID','histological_type']]
df_sample_type_squamous = df_sample_type[df_sample_type['histological_type'] == 'Esophagus Squamous Cell Carcinoma']
df_sample_type_adeno = df_sample_type[df_sample_type['histological_type'] == 'Esophagus Adenocarcinoma, NOS']
df_sample_type_squamous['histological_type'] = 0
df_sample_type_adeno['histological_type'] = 1
df_phenotype_prediction = pd.concat([df_sample_type_squamous,df_sample_type_adeno])
df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['histological_type'])
df_pca2_common.drop('histological_type',axis=1,inplace=True)


# In[63]:


l = ['EVC',
 'GCC2',
 'OR5T1',
 'AHNAK',
 'PFDN5',
 'C17orf69',
 'DHX30',
 'FASLG',
 'MAGEA2',
 'LOC146880']


# In[64]:


X_train,X_test,y_train,y_test = split_train_test(df_pca2_common,r)
X_train = X_train.reset_index(drop=True)
X_train=X_train[l]
X_test = X_test[l]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,Y_train.shape[0]]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,Z] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[65]:


sns.set_style('whitegrid')
dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-10,analysis_type)


# In[66]:


fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('ESCA subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig('str('baseline_filter_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[49]:


len(list_final_pred)/len(y_test)


# In[50]:


sns.set_style('darkgrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.1,fscore,label = 'BNN',color='r',width=0.05)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.05,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.1,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.05)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('ESCA types',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig(str('method_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# # Thyroid Carcinoma

# In[95]:


analysis_type = 'thca_subtypes'
cancer_types =['Papillary Carcinoma-Classical','Papillary Carcinoma-Follicular']


# In[96]:


df_pca2 = pd.read_csv('thyroid carcinoma',sep='\t')
df_samples = pd.DataFrame(list(df_pca2.columns)[1:])
df_samples.columns = ['sampleID']
df_pca2_new = df_pca2.iloc[:,1:].T.reset_index(drop=True)
genes = list(df_pca2['sample'])
df_pca2_new.columns = genes
df_pca2 = df_pca2_new
filelist = []
for file in glob.glob('Phenotypic_data1/*'):
    filelist.append(file)
c = 0    
for i in range(len(filelist)):
    if filelist[i][31:] =='thyroid carcinoma':
        print(i)
        c=i
        break
df_temp = pd.read_csv('Phenotypic_data1/'+str(filelist[c][31:]),sep='\t')
df_temp = df_samples.merge(df_temp,on='sampleID',how='inner')
df_sample_type = df_temp[['sampleID','histological_type']]
df_sample_type_papillary = df_sample_type[df_sample_type['histological_type'] == 'Thyroid Papillary Carcinoma - Classical/usual']
df_sample_type_follicular = df_sample_type[df_sample_type['histological_type'] == 'Thyroid Papillary Carcinoma - Follicular (>= 99% follicular patterned)']
df_sample_type_papillary['histological_type'] = 0
df_sample_type_follicular['histological_type'] = 1
df_phenotype_prediction = pd.concat([df_sample_type_papillary,df_sample_type_follicular])
df_sample_type_merged = df_samples.merge(df_phenotype_prediction,on='sampleID',how='inner')
df_pca2_with_samples = df_samples.merge(df_pca2,right_index=True,left_index=True)
df_pca2_common = df_sample_type_merged.merge(df_pca2_with_samples,on='sampleID',how='inner')
df_pca2_common.dropna(inplace=True)
df_pca2_common.drop('sampleID',axis=1,inplace=True)
df_pca2_common['label'] = list(df_pca2_common['histological_type'])
df_pca2_common.drop('histological_type',axis=1,inplace=True)


# In[97]:


l =['KLHDC10',
 'SBNO1',
 'ATE1',
 'ATF2',
 'CLCN3',
 'PTPRE',
 'FN1',
 'KCNQ3',
 'PROS1',
 'SLC34A2',
 'ARAP3',
 'GJC1',
 'GIPC3',
 'SPRY4',
 'MCAM',
 'RBM6',
 'KCNAB3',
 'ZNF700',
 'ANKLE1',
 'PVRIG',
 'EFHC1',
 'RMST',
 'CBX7',
 'CRTC1',
 'COTL1',
 'DHX38',
 'KIAA1967',
 'DDX23',
 'DHX37',
 'WDR81',
 'IFT57',
 'MRPL50',
 'INTS1',
 'GTF2B',
 'SSB',
 'NUPL2',
 'DDX55',
 'MYO19',
 'PIGL',
 'AGK',
 'TSPAN6',
 'FNTB',
 'APEX1',
 'ZBTB12',
 'RIC8A',
 'TINAGL1',
 'CD27',
 'HSPA12A',
 'FAM110B',
 'FBXW7']


# In[98]:


X_train,X_test,y_train,y_test = split_train_test(df_pca2_common,r)
X_train = X_train.reset_index(drop=True)
X_train=X_train[l]
X_test = X_test[l]
Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T
layers_dims = [X_train.shape[0],250,95,Y_train.shape[0]]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,Z] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[99]:


sns.set_style('whitegrid')
dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new = uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,1e-10,analysis_type)


# In[102]:


fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
plt.bar(np.arange(0,len(fscore)),fscore,label = 'BNN',color='magenta',width=0.1)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter-fscore,bottom = fscore,label = 'BNN+Filter',color='black',width=0.1)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('THCA subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1.45, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig(str('baseline_filter_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[73]:


len(list_final_pred)/len(y_test)


# In[74]:


sns.set_style('darkgrid')
fig = plt.figure()
fscore = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test))[2]
fscore_filter = precision_recall_fscore_support(np.array(list_final_test),np.array(list_final_pred))[2]
fscore_corrected = precision_recall_fscore_support(np.array(y_test),np.array(arr_pred_test_new))[2]
plt.bar(np.arange(0,len(fscore))-0.1,fscore,label = 'BNN',color='r',width=0.05)
plt.bar(np.arange(0,len(fscore_filter)),fscore_filter,width=0.05,label = 'BNN+Filter')
plt.bar(np.arange(0,len(fscore_corrected))+0.1,np.array(fscore_corrected),label='BNN+EpICC',color='g',width=0.05)
plt.ylabel('F1 Score',fontsize=25)
plt.xlabel('THCA subtypes',fontsize=25)
plt.xticks(np.arange(0,len(fscore)),cancer_types,fontsize=15)
plt.ylim((0.01,1.05))
plt.legend(bbox_to_anchor=(1, 1.02),prop={'size':15})
ax = plt.axes()
ax.yaxis.grid(True,which='major')
fig.savefig(str('method_comparison_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[ ]:


ep_test = com_ep_un(logits)
acc_2 = []
l_temp = np.arange(0.1,20.1,0.1)
sam_2 = []
for j in l_temp:
    list_final_pred = []
    list_final_test = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]*1e9 <= j* dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])
    acc_2.append(accuracy_score(list_final_test,list_final_pred))
    sam_2.append(len(list_final_test)*100/len(y_test))
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes()
ax2 = ax1.twinx()
l1 = ax1.plot(l_temp,np.array(acc_2),c='r',linestyle='-',label='Overall Accuracy')
#l2 = ax1.plot(l_temp,np.array(acc_1)*100,c='r',label='Overall Accuracy method 1')
ax1.tick_params(axis='y', labelcolor='r')
l2 = ax2.plot(l_temp,sam_2,c='b',linestyle = '-',label = 'No. of samples' )
#l4 = ax2.plot(l_temp,sam_1,c='g',label='No. of samples method 1')
ax2.tick_params(axis='y', labelcolor='b')
leg = l2+l1
ax1.set_ylabel('Overall Accuracy (%)', color='r',fontsize=15)
ax2.set_ylabel('% samples retained', color='b',fontsize=15)
ax1.set_xlabel('Filtering cut-off (Times train uncertainty)',fontsize=15)
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=(0.6,0.77))
ax2.set_yticks(np.arange(0,110,10))
ax1.set_xticks(np.arange(0,21))
ax1.grid()
fig.savefig(str('filtering_cutoff_')+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
plt.show()


# In[ ]:


fig = plt.figure()
plt.ylim(0.0,1.0)
plt.bar([1,2,3,4,5,6],[0.6061946902654868,0.4485981308411215,0.7077625570776256,0.6216216216216216,0.7,0.6571428571428571],width =0.2,color='black')
plt.xticks([1,2,3,4,5,6],['All cancer types', 'LGG subtypes','BRCA subtypes', 'COEN subtypes','ESCA subtypes','THCA sutypes'],rotation=30,fontsize=12)
plt.ylabel('% samples retained',fontsize=20)
fig.savefig('num_samples_retained.pdf', format='pdf', dpi=1200,bbox_inches='tight')


# # External Validation

# In[20]:


analysis_type='external_validation'
cancer_types = ['LAML','ACC','BLCA','LGG','BRCA','CESC','CHOL','COAD','UCEC','ESCA','GBM','HNSC','KIRC','KIRP','LIHC','LUAD','LUSC','DLBC','MESO','OV','PAAD','PCPG','PRAD','READ','SKCM','STAD','TGCT','THYM','THCA','UCS','UVM']
np.random.seed(0)
r = np.random.randint(500)
df_pca2 = pd.read_csv('100_genes_pca2_new.csv')
(X_train, X_test, y_train, y_test) = split_train_test(df_pca2,r)


# In[21]:


df_test = pd.read_csv('combined_external.csv')


# In[22]:


df_test = df_test[df_test['label'] == 4] 


# In[23]:


X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]


# In[24]:


Ytrain = np.array(y_train).reshape(len(y_train),)
X_train = np.array(X_train.T.reset_index(drop=True))
Y_train = tf.Session().run(tf.one_hot(Ytrain,len(set(Ytrain)),axis=1)).reshape(len(Ytrain),len(set(Ytrain))).T
Ytest = np.array(y_test).reshape(len(y_test),)
X_test = np.array(X_test.T.reset_index(drop=True))
Y_test = tf.Session().run(tf.one_hot(Ytest,len(set(Ytest)),axis=1)).reshape(len(Ytest),len(set(Ytest))).T


# In[25]:


layers_dims = [X_train.shape[0],250,95,31]
[_,_,logits,logits_train,arr_pred_test,arr_pred_train,grad_val] = model(X_train=X_train,Y_train=Y_train,X_test =X_test,Y_test = Y_test, learning_rate=0.0005,num_epochs=3500,print_cost=True,layers_dims = layers_dims)


# In[19]:


fig = plt.figure()
plt.ylim(80,100)
plt.bar([1,2,3],[90,100,100],width =0.2,color='lightblue')
plt.xticks([1,2,3],['BNN','BNN+Filter','BNN+EpICC'],fontsize=12)
plt.ylabel('Accuracy (%)',fontsize=20)
fig.savefig('external_validation.pdf', format='pdf', dpi=1200,bbox_inches='tight')

