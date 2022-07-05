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
 
    dict_mean_train = {}
    dict_mean_test_corr = {}
    dict_mean_test_incorr = {}
    for i in range(len(cancer_types)):
    
       
        train = dict_result_train[i][0]
        pred_train = dict_result_train[i][1]
        al_train_temp = dict_result_train[i][2]
        ep_train_temp = dict_result_train[i][3]
        
        
        
      
        
        list_index = list(np.where(arr_pred_test == i)[0])
        list_index2 = list(np.where(y_test==i)[0])
    
        correct_index = []
        incorrect_index = []
        for j in range(len(list_index)):
            if (list_index[j] in list_index2) == True:
                correct_index.append(j)
            else:
                incorrect_index.append(j)
        
        
        corr_al = al_test[list(np.array(list_index)[correct_index])]
        incorr_al = al_test[list(np.array(list_index)[incorrect_index])]
        
       
        list_combined = list(corr_al)+list(incorr_al)
        
        index_temp_incorr = []
        index_temp = []
        for j in range(len(list_combined)):
            index_temp.append(j)
            if (j >= len(corr_al)):
                index_temp_incorr.append(j)
        
        
        l_final, l_index_final = shuffle(list_combined, index_temp, random_state=0)
        
        
        
      
        
        incorrect = []
        for j in range(len(l_index_final)):
            if (l_index_final[j] in index_temp_incorr) ==True:
                incorrect.append(j)
        
        
        correct_train = []
        incorrect_train = []
        for k in range(len(list(pred_train))):
            if (pred_train[k] != train[k]):
                incorrect_train.append(k)
            else:
                correct_train.append(k)

        if uncertainty_type == 'Aleatoric':

            corr = np.mean(corr_al)
            incorr = np.mean(incorr_al)
            corr_train = np.mean(al_train_temp[correct_train])
            dict_mean_train[i] = corr_train
            dict_mean_test_corr[i] = corr
            dict_mean_test_incorr[i] = incorr
            
        if uncertainty_type == 'Epistemic':
            
            corr = np.mean(corr_al)*scal_fac
            incorr = np.mean(incorr_al)*scal_fac
            corr_train = np.mean(ep_train_temp[correct_train])*scal_fac
            dict_mean_train[i] = corr_train
            dict_mean_test_corr[i] = corr
            dict_mean_test_incorr[i] = incorr
            
       
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
        
        incorrect = []
        for i in range(len(list(arr_pred_test))):
            if (arr_pred_test[i] != np.array(y_test)[i]):
                incorrect.append(i)
                
        list_aleoteric_correct = []
        list_aleoteric_incorrect = []
        for i in range(len(aleoteric_uncertainty)):
            if (i in incorrect) == True:
                list_aleoteric_incorrect.append(aleoteric_uncertainty[i])
            if (i in incorrect) == False:
                list_aleoteric_correct.append(aleoteric_uncertainty[i])
                
        list_epistemic_correct = []
        list_epistemic_incorrect = []
        for i in range(len(epistemic_uncertainty)):
            if (i in incorrect) == True:
                list_epistemic_incorrect.append(epistemic_uncertainty[i])
            if (i in incorrect) == False:
                list_epistemic_correct.append(epistemic_uncertainty[i])
                
        corr = np.mean(np.array(list_aleoteric_correct))
        incorr = np.mean(np.array(list_aleoteric_incorrect))
        corr_epistemic = np.mean(np.array(list_epistemic_correct))
        incorr_epistemic = np.mean(np.array(list_epistemic_incorrect))
        
        if type_uncer == 'Aleatoric':
            fig = plt.figure()
            ax1 = fig.add_axes([0, 0, 1, 1])
            ax2 = fig.add_axes()
            ax2 = ax1.twinx()
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')

            
            sns.kdeplot(list_aleoteric_correct,color='b',ax=ax1)
            sns.kdeplot(list_aleoteric_incorrect,color='r',ax=ax2).set(xlim=(0, 0.35))
   
            if t == 'Train':
                plt.axvline(x=corr,c= 'b',linestyle='--',label = 'Mean uncertainty (correct)')
                plt.axvline(x=incorr, c = 'r',linestyle='--',label = 'Mean uncertainty (incorrect)')
                
           
            
            if t == 'Validation':
                plt.axhline(y=corr,c= 'g',label='Mean uncertainty for correct predictions')
                plt.axhline(y=incorr, c = 'm',label='Mean uncertainty for incorrect predictions')
                plt.legend(bbox_to_anchor=(1,1),prop={'size':10})
   
            if t== 'Test':
                plt.axvline(x=corr, c = 'b',linestyle='--',label= 'Mean test uncertainty (correct)')
                plt.axvline(x=incorr,c= 'r',linestyle='--',label = 'Mean test uncertaity (incorrect)')
                plt.axvline(x=t_l, c = 'g',linestyle='--',label = 'Mean train uncertainty (correct)')
                
            plt.legend(bbox_to_anchor=(1.6,1))    
            
                
            ax1.set_ylabel('Density (Correct Predictions)', color='b',fontsize=15)
            ax2.set_ylabel('Density (Incorrect Predictions)', color='r',fontsize=15)

            ax1.set_xlabel('Aleatoric Uncertainty',fontsize = 15)
           
            plt.show()
            return [aleoteric_uncertainty,corr,incorr]

        if type_uncer == 'Epistemic':
            fig = plt.figure()
            ax1 = fig.add_axes([0, 0, 1, 1])
            ax2 = fig.add_axes()
            ax2 = ax1.twinx()
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
           
            
            sns.kdeplot(np.array(list_epistemic_correct)/scal_fac,color='b',ax=ax1)
            sns.kdeplot(np.array(list_epistemic_incorrect)/scal_fac,color='r',ax=ax2).set(xlim=(0, 2))
   
            if t == 'Train':
                plt.axvline(x=corr_epistemic/scal_fac, c = 'b',linestyle='--',label = 'Mean uncertainty (correct)')
                plt.axvline(x=incorr_epistemic/scal_fac,c= 'r',linestyle='--',label = 'Mean uncertainty (incorrect)')
                
   
            if t== 'Test':
                plt.axvline(x=corr_epistemic/scal_fac, c = 'b',linestyle='--',label = 'Mean test uncertainty (correct)')
                plt.axvline(x=incorr_epistemic/scal_fac,c= 'r',linestyle='--',label = 'Mean test uncertainty (incorrect)')
                plt.axvline(x=t_l*scal_fac, c = 'g',linestyle='--',label='Mean train uncertainty (correct)')
                
                
            
            plt.legend(bbox_to_anchor=(1.6,1))
            
            ax1.set_ylabel('Density (Correct Predictions)', color='b',fontsize=15)
            ax2.set_ylabel('Density (Incorrect Predictions)', color='r',fontsize=15)
            ax1.set_xlabel('Epistemic Uncertainty (x '+str(r'$10^{'+str(scal_fac).split('e')[-1]+'}$')+')',fontsize = 15)
            
            fig.savefig('Desktop/paper_revision/'+str(image_name)+'.pdf', format='pdf', dpi=1200,bbox_inches='tight')
            plt.show()
            return [epistemic_uncertainty,corr_epistemic,incorr_epistemic]


# In[5]:




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
