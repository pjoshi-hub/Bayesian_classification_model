# EpICC: A Bayesian Neural Network with uncertainty correction for accurate classification of cancer 

**Overall workflow**

![alt text](https://github.com/pjoshi-hub/Bayesian_classification_model/blob/main/Figures/uncertainty_workflow.jpg)



# <sub><sup>Implementation</sup><sub>

**.py files**
 - BNN.py:  Bayesian Neural Network
 - uncertainty.py : code to calculate and correct uncertainty
 - DNN_classification.py: Deep Neural Network based on point estimates
 - Feature_selection_and_PCA_plots.py: code to select gene set from pca
 - implementation.py: code to implement and generate results
 
 implementation.py contains the coded to generate results and the individual .py files represent the different code blocks used in implementation.py
 
 **.ipynb files**
- Feature_selection_and_PCA_plots.ipynb: step by step Jupyter notebook execution to select gene set
- Cancer_classification.ipynb: step by step Jupyter notebook execution to reproduce results for cancer classification reported in the study
- Cancer_subtype_classifcation: step by step Jupyter notebook execution to reprodce results for cancer subtype classifcation reported in the study
 
 The datasets can be found at: https://drive.google.com/drive/folders/1afzlMAiHy3LWFoA-M5N4OUDMaMa70o3S?usp=sharing
 
 To successfully execute the code:
 1. The datasets sould be loaded into the current working directory. This is a prerequisite to run all the codes.
 2. The code blocks in the .ipynb files should be executed. The naming convention of the datasets in the link is the same as that mentioned in the codes.
 3. To generate additional analysis figures in the study, the folder "codes" should be cloned to the default directory of the jupyter notebook and the following line should be run:
```
%run implementation.py
```
 
 
