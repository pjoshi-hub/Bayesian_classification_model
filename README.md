# EpICC: A Bayesian Neural Network with uncertainty correction for accurate classification of cancer 

**Overall workflow**

![alt text](https://github.com/pjoshi-hub/Bayesian_classification_model/blob/main/Figures/uncertainty_workflow.jpg)




**Uncertainty correction can be used to improve the performance of prediction models**


# <sub><sup>Implementation</sup><sub>

**.py files**
 - BNN.py:  Bayesian neural network
 - uncertainty.py : code to calculate and correct uncertainty
 - DNN_classification.py: code to train DNN and test
 - Feature_selection_and_PCA_plots.py: code to select gene set from pca
 - Implementation.py: code to implement and generate results
 
 **.ipynb files**
- Feature_selection_and_PCA_plots.ipynb: step by step execution to select gene set
- Cancer_classification.ipynb: step by step execution to reproduce results for cancer classification reported in the study
- Cancer_subtype_classifcation: step by step execution to reprodce results for cancer subtype classifcation reported in the study
 
 The datasets can be found at: https://drive.google.com/drive/folders/1afzlMAiHy3LWFoA-M5N4OUDMaMa70o3S?usp=sharing
 
 To successfully execute the code:
 1. The datasets sould be loaded into the current working directory
 2. The code blocks should be executed. The naming convention of the datasets in the link is the same that is mentioned in the link.
 3. To generate the final results, the .ipynb files should be run after loading the datasets mentioned in the link into the present working directory. 
 
 To generate the analysis figures in the study, the Implementation.py file should be run after loading the datasets mentioned in the link into the present working directory. .
