# EpICC: A Bayesian Neural Network with uncertainty correction for accurate classification of cancer 

**Overall workflow**

![alt text](https://github.com/pjoshi-hub/Bayesian_classification_model/blob/main/Figures/uncertainty_workflow.jpg)



# <sub><sup>1: Gene set selection</sup></sub>

![alt text](https://github.com/pjoshi-hub/Bayesian_classification_model/blob/main/Figures/Feature_selection_pca2.JPG)



**Uncertainty correction can be used to improve the performance of prediction models**


# <sub><sup>Implementation</sup><sub>

**Codes and Implementation**
 - BNN.py:  Bayesian neural network
 - uncertainty.py : code to calculate and correct uncertainty
 - DNN_classification.py: code to train DNN and test
 - Feature_selection_and_PCA_plots.py: code to select gene set from pca
 - Implementation.py: code to implement and generate analysis and results
 
 **Ipython notebooks**
- Feature_selection_and_PCA_plots.ipynb: step by step execution to select gene set
- Cancer_classification.ipynb: step by step execution to reproduce results for cancer classification reported in the study
- Cancer_subtype_classifcation: step by step execution to reprodce results for cancer subtype classifcation reported in the study
 
 The datasets can be found at: 
 
 To successfully execute the code, the dataset sould be loaded into the current working directory, and the code blocks should be executed. The naming convention of the datasets in the link is the same that is mentioned in the link.
