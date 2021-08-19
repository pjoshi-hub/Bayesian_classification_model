
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


# In[2]:


df_label = pd.read_csv('D:/New Folder/labels.csv',header=1)
label = list(df_label['Abbreviation'])


# In[3]:


def detailed_plot_new(dict_result_train,uncertainty_type,y_test,arr_pred_test,al_test,image_name,cancer_types,scal_fac=1e9):
    #fig = plt.figure(figsize = (20,20))
    #fig.text(0.5, 0.04, '', ha='center', va='center',fontsize=30)
    #if uncertainty_type == 'Aleatoric':
        #fig.text(0.06, 0.5, 'Aleatoric Uncertainty', ha='center', va='center', rotation='vertical',fontsize=30)

    #if uncertainty_type == 'Epistemic':
        #fig.text(0.06, 0.5, 'Epistemic Uncertainty (x '+str(r'$10^{-9}$')+')', ha='center', va='center', rotation='vertical',fontsize=30)

    #cancer_types = ['LAML','ACC','BLCA','LGG','BRCA','CESC','CHOL','COAD','UCEC','ESCA','GBM','HNSC','KIRC','KIRP','LIHC','LUAD','LUSC','DLBC','MESO','OV','PAAD','PCPG','PRAD','READ','SKCM','STAD','TGCT','THYM','THCA','UCS','UVM']
    dict_mean_train = {}
    dict_mean_test_corr = {}
    dict_mean_test_incorr = {}
    for i in range(len(cancer_types)):
    
        #test = dict_result_test[i][0]
        #pred = dict_result_test[i][1]
        #al = dict_result_test[i][2]
        #ep = dict_result_test[i][3]
        train = dict_result_train[i][0]
        pred_train = dict_result_train[i][1]
        al_train_temp = dict_result_train[i][2]
        ep_train_temp = dict_result_train[i][3]
        
        
        
        #train = dict_result_train[i][2]
        #a = np.mean(dict_prob_temp[i],axis=0)
        #b = np.std(dict_prob_temp[i],axis=0) * 1e3 * 2
        #x = np.arange(0,len(test))
        #ax = fig.add_subplot(7,5,i+1)
        #plt.errorbar(x, a, b, linestyle='None',marker='.',ecolor='r',c='g',markeredgewidth=0.00001)
        
        list_index = list(np.where(arr_pred_test == i)[0])
        list_index2 = list(np.where(y_test==i)[0])
    
        correct_index = []
        incorrect_index = []
        for j in range(len(list_index)):
            if (list_index[j] in list_index2) == True:
                correct_index.append(j)
            else:
                incorrect_index.append(j)
        #print(correct_index)
        
        corr_al = al_test[list(np.array(list_index)[correct_index])]
        incorr_al = al_test[list(np.array(list_index)[incorrect_index])]
        
        #corr_al+incorr_al
        list_combined = list(corr_al)+list(incorr_al)
        
        index_temp_incorr = []
        index_temp = []
        for j in range(len(list_combined)):
            index_temp.append(j)
            if (j >= len(corr_al)):
                index_temp_incorr.append(j)
        #print(index_temp_incorr)
                
        #c = zip(list_combined,index_temp)
        
        l_final, l_index_final = shuffle(list_combined, index_temp, random_state=0)
        
        
        
        #l_final = [e[0] for e in c]
        #l_index_final = [e[1] for e in c]
        
        uncorrect = []
        for j in range(len(l_index_final)):
            if (l_index_final[j] in index_temp_incorr) ==True:
                uncorrect.append(j)
        #print(uncorrect)      
            
        
        correct_train = []
        uncorrect_train = []
        for k in range(len(list(pred_train))):
            if (pred_train[k] != train[k]):
                uncorrect_train.append(k)
            else:
                correct_train.append(k)

        if uncertainty_type == 'Aleatoric':

            corr = np.mean(corr_al)
            uncorr = np.mean(incorr_al)
            corr_train = np.mean(al_train_temp[correct_train])
            #bar_list = ax.bar(np.arange(0,len(l_final)),l_final,width=0.1)
            dict_mean_train[i] = corr_train
            dict_mean_test_corr[i] = corr
            dict_mean_test_incorr[i] = uncorr
            
        if uncertainty_type == 'Epistemic':
            
            corr = np.mean(corr_al)*scal_fac
            uncorr = np.mean(incorr_al)*scal_fac
            corr_train = np.mean(ep_train_temp[correct_train])*scal_fac
            #bar_list = ax.bar(np.arange(0,len(l_final)),np.array(l_final)*scal_fac,width=0.1)
            dict_mean_train[i] = corr_train
            dict_mean_test_corr[i] = corr
            dict_mean_test_incorr[i] = uncorr
            
        #for j in range(len(uncorrect)):
            #bar_list[uncorrect[j]].set_color('r')
        #ax.axhline(y=corr_train, c = 'g')
        #ax.axhline(y=corr, c = 'y')
        #ax.axhline(y=uncorr, c = 'm')
        #plt.axhline(y=uncorr_epistemic,c= 'm')
        #plt.ylabel('Epistemic Uncertainty (x '+str(r'$10^{-9}$')+')',fontsize = 15)
        #plt.xlabel('Samples',fontsize=15)
        #plt.show()

        #ax.set_title(cancer_types[i]+' ('+str(len(l_final))+')')
        #ax.set_ylim(0,1.5)
        #ax.set_xticklabels('')
    #fig.savefig('Desktop/paper_figures/'+str(image_name)+'.pdf', format='pdf', dpi=120,bbox_inches='tight')
    return dict_mean_train,dict_mean_test_corr,dict_mean_test_incorr


# In[2]:


def uncertainty_calculation(logits,y_test,arr_pred_test,type_uncer,t,t_l,image_name,scal_fac=1e-9):
    
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
        aleoteric_uncertainty = np.mean(np.array(aleo_list),axis=0)
        
        uncorrect = []
        for i in range(len(list(arr_pred_test))):
            if (arr_pred_test[i] != np.array(y_test)[i]):
                uncorrect.append(i)
                
        list_aleoteric_correct = []
        list_aleoteric_incorrect = []
        for i in range(len(aleoteric_uncertainty)):
            if (i in uncorrect) == True:
                list_aleoteric_incorrect.append(aleoteric_uncertainty[i])
            if (i in uncorrect) == False:
                list_aleoteric_correct.append(aleoteric_uncertainty[i])
                
        list_epistemic_correct = []
        list_epistemic_incorrect = []
        for i in range(len(epistemic_uncertainty)):
            if (i in uncorrect) == True:
                list_epistemic_incorrect.append(epistemic_uncertainty[i])
            if (i in uncorrect) == False:
                list_epistemic_correct.append(epistemic_uncertainty[i])
                
        corr = np.mean(np.array(list_aleoteric_correct))
        uncorr = np.mean(np.array(list_aleoteric_incorrect))
        corr_epistemic = np.mean(np.array(list_epistemic_correct))
        uncorr_epistemic = np.mean(np.array(list_epistemic_incorrect))
        
        if type_uncer == 'Aleatoric':
            fig = plt.figure()
            ax1 = fig.add_axes([0, 0, 1, 1])
            ax2 = fig.add_axes()
            ax2 = ax1.twinx()
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')

            #bar_list = plt.bar(np.arange(0,len(aleoteric_uncertainty)),aleoteric_uncertainty,label='Correct Predictions')
            #for i in range(len(uncorrect)):
                #if (i == len(uncorrect)-1):
                    #bar_list[uncorrect[i]].set_color('r')
                    #bar_list[uncorrect[i]].set_label('Incorrect Predictions')
                #else:
                    #bar_list[uncorrect[i]].set_color('r')
            #plt.xticks([])
            sns.kdeplot(list_aleoteric_correct,color='b',ax=ax1)
            sns.kdeplot(list_aleoteric_incorrect,color='r',ax=ax2).set(xlim=(0, 0.35))
   
            if t == 'Train':
                plt.axvline(x=corr,c= 'b',linestyle='--',label = 'Mean uncertainty (correct)')
                plt.axvline(x=uncorr, c = 'r',linestyle='--',label = 'Mean uncertainty (incorrect)')
                
            #plt.axhline(y=corr, c = 'y')
            
            if t == 'Validation':
                plt.axhline(y=corr,c= 'g',label='Mean uncertainty for correct predictions')
                plt.axhline(y=uncorr, c = 'm',label='Mean uncertainty for incorrect predictions')
                plt.legend(bbox_to_anchor=(1,1),prop={'size':10})
   
            if t== 'Test':
                plt.axvline(x=corr, c = 'b',linestyle='--',label= 'Mean test uncertainty (correct)')
                plt.axvline(x=uncorr,c= 'r',linestyle='--',label = 'Mean test uncertaity (incorrect)')
                plt.axvline(x=t_l, c = 'g',linestyle='--',label = 'Mean train uncertainty (correct)')
                
            plt.legend(bbox_to_anchor=(1.6,1))    
            #labs = [l.get_label() for l in leg]
            #ax1.legend(leg, labs, loc=(1,1))
                
            ax1.set_ylabel('Density (Correct Predictions)', color='b',fontsize=15)
            ax2.set_ylabel('Density (Incorrect Predictions)', color='r',fontsize=15)

            ax1.set_xlabel('Aleatoric Uncertainty',fontsize = 15)
            #plt.xlabel('Aleatoric Uncertainty',fontsize = 15)
            #plt.ylabel('Samples',fontsize=15)
            #fig.savefig('Desktop/paper_revision/'+str(image_name)+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
            plt.show()
            return [aleoteric_uncertainty,corr,uncorr]

        if type_uncer == 'Epistemic':
            fig = plt.figure()
            ax1 = fig.add_axes([0, 0, 1, 1])
            ax2 = fig.add_axes()
            ax2 = ax1.twinx()
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            #bar_list = plt.bar(np.arange(0,len(epistemic_uncertainty)),epistemic_uncertainty*scal_fac)
            #for i in range(len(uncorrect)):
                #bar_list[uncorrect[i]].set_color('r')
            #plt.xticks([])
            
            sns.kdeplot(np.array(list_epistemic_correct)/scal_fac,color='b',ax=ax1)
            sns.kdeplot(np.array(list_epistemic_incorrect)/scal_fac,color='r',ax=ax2).set(xlim=(0, 2))
   
            if t == 'Train':
                plt.axvline(x=corr_epistemic/scal_fac, c = 'b',linestyle='--',label = 'Mean uncertainty (correct)')
                plt.axvline(x=uncorr_epistemic/scal_fac,c= 'r',linestyle='--',label = 'Mean uncertainty (incorrect)')
                #leg = l1+l2
                #plt.axhline(y=corr, c = 'y')
   
            if t== 'Test':
                plt.axvline(x=corr_epistemic/scal_fac, c = 'b',linestyle='--',label = 'Mean test uncertainty (correct)')
                plt.axvline(x=uncorr_epistemic/scal_fac,c= 'r',linestyle='--',label = 'Mean test uncertainty (incorrect)')
                plt.axvline(x=t_l*scal_fac, c = 'g',linestyle='--',label='Mean train uncertainty (correct)')
                
                
            #labs = [l.get_label() for l in leg]
            plt.legend(bbox_to_anchor=(1.6,1))
            
            ax1.set_ylabel('Density (Correct Predictions)', color='b',fontsize=15)
            ax2.set_ylabel('Density (Incorrect Predictions)', color='r',fontsize=15)
            ax1.set_xlabel('Epistemic Uncertainty (x '+str(r'$10^{'+str(scal_fac).split('e')[-1]+'}$')+')',fontsize = 15)
            #plt.axhline(y=corr_epistemic, c = 'g')
            #plt.axhline(y=uncorr_epistemic,c= 'm')
            #plt.xlabel('Epistemic Uncertainty (x '+str(r'$10^{-9}$')+')',fontsize = 15)
            #plt.ylabel('Samples',fontsize=15)
            fig.savefig('Desktop/paper_revision/'+str(image_name)+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
            plt.show()
            return [epistemic_uncertainty,corr_epistemic,uncorr_epistemic]


# In[5]:


def com_al_un(logits,arg_passed='argmax'):
    aleo_list = []
    for j in range(len(logits)):
        prob_list = []
        for i in range(logits[j].shape[1]):
            if arg_passed=='argmax':
                arg = np.argmax(logits[j][:,i])
            else:
                arg = arg_passed   
            prob = logits[j][:,i][arg]
            prob_list.append(prob)
        aleo = list(np.array(prob_list) - np.square(np.array(prob_list)))
        aleo_list.append(aleo)
        
    aleatoric_uncertainty = np.mean(np.array(aleo_list),axis=0)
    
    return aleatoric_uncertainty

def com_ep_un(logits,arg_passed='argmax'):        
    epi_list = []
    for j in range(len(logits)):
        prob_list = []
        for i in range(logits[j].shape[1]):
            if arg_passed=='argmax':
                arg = np.argmax(logits[j][:,i])
            else:
                arg = arg_passed
            prob = logits[j][:,i][arg]
            prob_list.append(prob)
        epi_list.append(np.array(prob_list))
        
    epistemic_uncertainty = np.mean(np.square((np.array(epi_list)-np.mean(np.array(epi_list),axis=0))),axis=0)
    
    return epistemic_uncertainty


# In[2]:


def uncertainty_correction(test_logits):
    
    adjusted_logits_complete = []
    mean_test_logits = sum(test_logits)/len(test_logits)
    func_mean_test_logits = np.log(np.array(mean_test_logits)/(1-np.array(mean_test_logits)))
    
    
    for index in range(test_logits[0].shape[0]):
    
        #epistemic_test = np.sqrt(com_ep_un(test_logits,index))
        #epistemic_test_norm = epistemic_test/np.max(epistemic_test)
        epistemic_test = np.sqrt(com_ep_un(test_logits,index))
        epistemic_test_norm = epistemic_test/np.max(epistemic_test)

        X = epistemic_test_norm
        Y = list(func_mean_test_logits[index,:])
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        adj_logits = func_mean_test_logits[index,:]-model.params[1]*epistemic_test_norm
        func_inv_adj_logits = np.exp(adj_logits)/(1+np.exp(adj_logits))
        adjusted_logits_complete.append(func_inv_adj_logits)

    return adjusted_logits_complete


# In[ ]:


def uncertainty(test_logits):
    
    adjusted_logits_complete = []
    #mean_test_logits = sum(Z)/len(Z)
    #func_mean_test_logits = mean_test_logits
    #mean_train_logits = sum(train_logits)/len(logits)
    mean_test_logits = sum(test_logits)/len(test_logits)
    func_mean_test_logits = np.log(np.array(mean_test_logits)/(1-np.array(mean_test_logits)))
    #func_mean_test_logits = mean_test_logits
    
    #y_train_ols = []
    
    #for i in range(len(y_train)):
        #if np.argmax(mean_train_logits)[:,i] == y_test[i]:
           # y_train_ols.append(1)
        #else:
            #y_train_ols.appendd(0)
            
    #y_test_ols = []
    #for i in range(len(y_test)):
        #if np.argmax(mean_test_logits)[:,i] == y_test[i]:
            #y_test_ols.append(1)
        #else:
            #y_test_ols.append(0)
    
    
    for index in range(3):
    
        #aleatoric_test = np.sqrt(com_al_un(test_logits,index))
        #aleatoric_test_norm = aleatoric_test/np.max(aleatoric_test)
        epistemic_test = np.sqrt(com_ep_un(test_logits,index))
        epistemic_test_norm = epistemic_test/np.max(epistemic_test)

        X = epistemic_test_norm
        #X = np.vstack((aleatoric_test_norm,epistemic_test_norm)).T
        Y = list(func_mean_test_logits[index,:])
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        #adj_logits = func_mean_test_logits[index,:]-model.params[1]*aleatoric_test_norm-model.params[2]*epistemic_test_norm
        adj_logits = func_mean_test_logits[index,:]-model.params[1]*epistemic_test_norm
        func_inv_adj_logits = np.exp(adj_logits)/(1+np.exp(adj_logits))
        #func_inv_adj_logits = adj_logits
        adjusted_logits_complete.append(func_inv_adj_logits)

    return adjusted_logits_complete,model.params


# def uncertainty_correction(test_logits,y_test):
#     
#     adjusted_logits_complete = []
#     #mean_test_logits = sum(Z)/len(Z)
#     #func_mean_test_logits = mean_test_logits
#     #mean_train_logits = sum(train_logits)/len(logits)
#     mean_test_logits = sum(test_logits)/len(test_logits)
#     func_mean_test_logits = np.log(np.array(mean_test_logits)/(1-np.array(mean_test_logits)))
#     #func_mean_test_logits = mean_test_logits
#     
#     #y_train_ols = []
#     
#     #for i in range(len(y_train)):
#         #if np.argmax(mean_train_logits)[:,i] == y_test[i]:
#            # y_train_ols.append(1)
#         #else:
#             #y_train_ols.appendd(0)
#             
#     #y_test_ols = []
#     #for i in range(len(y_test)):
#         #if np.argmax(mean_test_logits)[:,i] == y_test[i]:
#             #y_test_ols.append(1)
#         #else:
#             #y_test_ols.append(0)
#     
#     
#     for index in range(31):
#     
#         aleatoric_test = com_al_un(test_logits,index)
#         aleatoric_test_norm = aleatoric_test/np.max(aleatoric_test)
#         #epistemic_test = np.sqrt(com_ep_un(test_logits,index))
#         #epistemic_test_norm = epistemic_test/np.max(epistemic_test)
# 
#         #X = epistemic_test_norm
#         X = (aleatoric_test_norm).T
#         Y = list(func_mean_test_logits[index,:])
#         X = sm.add_constant(X)
#         model = sm.OLS(Y, X).fit()
#         #adj_logits = func_mean_test_logits[index,:]-model.params[1]*aleatoric_test_norm-model.params[2]*epistemic_test_norm
#         #adj_logits = func_mean_test_logits[index,:]-model.params[1]*aleatoric_test_norm-model.params[2]*epistemic_test_norm
#         adj_logits = func_mean_test_logits[index,:]-model.params[1]*aleatoric_test_norm
#         func_inv_adj_logits = np.exp(adj_logits)/(1+np.exp(adj_logits))
#         #func_inv_adj_logits = adj_logits
#         adjusted_logits_complete.append(func_inv_adj_logits)
# 
#     return adjusted_logits_complete

# def uncertainty(test_logits,list_params='Train'):
#     epistemic_list =[]
#     
#     adjusted_logits_complete = []
#     #mean_test_logits = sum(Z)/len(Z)
#     #func_mean_test_logits = mean_test_logits
#     #mean_train_logits = sum(train_logits)/len(logits)
#     mean_test_logits = sum(test_logits)/len(test_logits)
#     func_mean_test_logits = np.log(np.array(mean_test_logits)/(1-np.array(mean_test_logits)))
#     #func_mean_test_logits = mean_test_logits
#     
#     #y_train_ols = []
#     
#     #for i in range(len(y_train)):
#         #if np.argmax(mean_train_logits)[:,i] == y_test[i]:
#            # y_train_ols.append(1)
#         #else:
#             #y_train_ols.appendd(0)
#             
#     #y_test_ols = []
#     #for i in range(len(y_test)):
#         #if np.argmax(mean_test_logits)[:,i] == y_test[i]:
#             #y_test_ols.append(1)
#         #else:
#             #y_test_ols.append(0)
#     
#     
#     #for index in range(3):
#     list_model_params = []
#     
#     if list_params == 'Train':
#         
#         for index in range(test_logits[0].shape[0]):
#     #aleatoric_test = np.sqrt(com_al_un(test_logits,index))
#     #aleatoric_test_norm = aleatoric_test/np.max(aleatoric_test)
#         
#         #aleatoric_test = np.sqrt(com_al_un(test_logits,index))
#             epistemic_test = np.sqrt(com_ep_un(test_logits,index))
#             #max_al = np.max(aleatoric_test)
#             #max_ep = np.max(epistemic_test)
#             #aleatoric_test_norm = aleatoric_test/max_al
#             epistemic_test_norm = epistemic_test
# 
#             X = epistemic_test_norm
#             #X = np.vstack((aleatoric_test_norm,epistemic_test_norm)).T
#             Y = list(func_mean_test_logits[index,:])
#             X = sm.add_constant(X)
#             model = sm.OLS(Y, X).fit()
#             adj_logits = func_mean_test_logits[index,:]-model.params[1]*epistemic_test_norm
#             list_model_params.append(model.params[1])
#             #model_params = get_params(epistemic_test_norm,Y)
#             #adj_logits = func_mean_test_logits[index,:]-epistemic_test_norm*model.params[1]
#             #func_inv_adj_logits = np.exp(adj_logits)/(1+np.exp(adj_logits))
#             #func_inv_adj_logits = adj_logits
#             #adjusted_logits_complete.append(func_inv_adj_logits)
# 
#         return epistemic_test,list_model_params 
#         
#     else:
#         
#         for index in range(test_logits[0].shape[0]):
#     
#         #aleatoric_test = np.sqrt(com_al_un(test_logits,index))
#         #aleatoric_test_norm = aleatoric_test/np.max(aleatoric_test)
#             epistemic_test = np.sqrt(com_ep_un(test_logits,index))
#             #func_inv_adj_logits = adj_logits
#             epistemic_list.append(list(epistemic_test))
# 
# 
# 
#         return epistemic_list

# In[1]:


def uncertainty_analysis(logits_train,y_train,arr_pred_train,logits,y_test,arr_pred_test,scal_fac,analysis_type):
    [al_train,mu_train_ac,mu_train_nac] = uncertainty_calculation(logits_train,y_train,arr_pred_train,'Aleatoric','Train',_,analysis_type)
    [ep_train,mu_train_ec,mu_train_nec] = uncertainty_calculation(logits_train,y_train,arr_pred_train,'Epistemic','Train',_,analysis_type,scal_fac)
    [al_test,mu_test_ac,mu_test_nac] = uncertainty_calculation(logits,y_test,arr_pred_test,'Aleatoric','Test',mu_train_ac,analysis_type)
    [ep_test,mu_test_ec,mu_test_nec] = uncertainty_calculation(logits,y_test,arr_pred_test,'Epistemic','Test',mu_train_ec,analysis_type,scal_fac)
    dict_numtrain_samples = Counter(np.array(y_train))
    dict_numtest_samples = Counter(np.array(y_test))
    dict_result_train = type_wise_results(dict_numtrain_samples,np.array(y_train),arr_pred_train,al_train,ep_train)
    dict_result_test = type_wise_results(dict_numtest_samples,np.array(y_test),arr_pred_test,al_test,ep_test)
    dict_result_train = type_wise_results(dict_numtrain_samples,np.array(y_train),arr_pred_train,al_train,ep_train)
    dict_result_test = type_wise_results(dict_numtest_samples,np.array(y_test),arr_pred_test,al_test,ep_test)
    #dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr = cancer_wise_uncertainty(dict_result_train,'Epistemic',np.array(y_test),arr_pred_test,ep_test,'epistemic_uncertainty_cancer_wise')
    dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr = detailed_plot_new(dict_result_train,'Epistemic',np.array(y_test),arr_pred_test,ep_test,'epistemic_uncertainty_cancer_wise',cancer_types,1/scal_fac)
    list_final_pred = []
    list_final_test = []
    uncertain = []
    certain = []
    for i in range(len(arr_pred_test)):
        if (ep_test[i]/scal_fac <= dict_mean_ep[arr_pred_test[i]]):
            list_final_test.append(np.array(y_test)[i])
            list_final_pred.append(arr_pred_test[i])

            certain.append(i)

        else:
            uncertain.append(i)
            
    start_index = 0
    for i in range(logits[0].shape[0]):
        print(i)
        temp_index = Counter(y_test)[i]
        end_index = start_index+temp_index
        logits_temp = list(np.array(logits)[:,:,start_index:end_index])
        adjusted_logits_temp = uncertainty_correction(logits_temp)
        print(np.array(adjusted_logits_temp).shape)
        if i == 0:
            adjusted_logits_complete = np.array(adjusted_logits_temp)
        else:
            adjusted_logits_complete = np.concatenate((adjusted_logits_complete,np.array(adjusted_logits_temp)),axis=1)
        start_index = end_index
    res = adjusted_logits_complete
    arr_pred_test_new = []
    counter = 0
    for i in range(logits[0].shape[1]):
        if np.argmax(np.array(res)[:,i]) == list(y_test)[i]:
            counter = counter+1
        arr_pred_test_new.append(np.argmax(np.array(res)[:,i]))
    #counter/len(arr_pred_test_new)
    print('EpICC\n')
    print(accuracy_score(y_test,arr_pred_test_new))
    print(np.mean(precision_recall_fscore_support(y_test,arr_pred_test_new)[1]))
    print('Filtering\n')
    print(accuracy_score(list_final_test,list_final_pred))
    print(np.mean(precision_recall_fscore_support(list_final_test,list_final_pred)[1]))
    
    fig = plt.figure()
    for key in dict_mean_test_corr.keys():
        if analysis_type != 'lgg_subtypes':
            plt.bar(key-0.1,dict_mean_ep[key],width=0.05,color='green')
            plt.bar(key,dict_mean_test_corr[key],width=0.05,color='blue')
            plt.bar(key+0.1,dict_mean_test_incorr[key],width=0.05,color='red')
        else:
            plt.bar(key-0.2,dict_mean_ep[key],width=0.1,color='green')#,label='Correct train')
            plt.bar(key,dict_mean_test_corr[key],width=0.1,color='blue')#,label='Correct test')
            plt.bar(key+0.2,dict_mean_test_incorr[key],width=0.1,color='red')#,label = 'Incorrect test')
    plt.ylabel('Mean Uncertainty (x'+str(r'$10^{'+str(scal_fac).split('e')[-1]+'}$')+')',fontsize = 15)
    plt.xlabel(analysis_type.split('_')[0].upper()+' '+analysis_type.split('_')[1],fontsize=20)
    plt.xticks(np.arange(0,len(cancer_types)),cancer_types,fontsize=15)
    plt.legend(prop={'size':12})
    ax = plt.gca()
    ax.yaxis.grid(True,which='major')
    #ax.yaxis.grid(True,which='minor',linestyle='--')
    fig.savefig('Desktop/paper_revision/uncert_comparison_'+analysis_type+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
    plt.show()
    
    return dict_mean_ep,dict_mean_test_corr,dict_mean_test_incorr,res,list_final_test,list_final_pred,arr_pred_test_new

