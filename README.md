# mortgage_default_model

## Introduction

This is a project to compare various models for predicting serious 
mortgage default. The entry level model is a multivariate logistic 
regression model. In addition, we implemented a KNN model, and a 
more complex neural network. We find that, with the data used for
the task, more advanced neural network topologies have not led to
a substantial increase in prediction. With that in mind, other texts
have indicated that, with additional data, it is possible to predict
non linear decision boundaries with deep learning. 

## Data

For this project, we are using FNMA's Single Family Loan Performance
Data. FNMA publishes quarterly data for each year. The data consists
of two files - one for acquisitions, and one for performance. The 
acquisition file contains information on a loan at the time it was
originated. The performance file has a monthly history of loan 
performance. For this project, we are using Q4 2007. 

While there are additional sources of data, such as FHA and FHLMC, 
we choose to work with this mostly homogeneous data set. 

### Transformations

All data transformations are maintained in data_transform.py. From
the performance file, we compute a few features - only two of which
are used. We have the number of times a loan was 0, 1, 2, and 3 
payments due. We've also computed whether the loan went to 
foreclosure, and the number of reporting periods the loan is
present. Note that a loan going to foreclosure does not necessarily
mean that the property was sold, or deeded back to the investor. 
The foreclosure status is computed based on the date the first
legal action occurs, at which point, a borrower can still reinstate.

Once the features are extracted from the performance file, they are
joined back to the acquisition data, yielding one record for each
loan. 

For the predictors chosen in the final model, we impute the mean over
nulls.

## Model

### Overview

We've chosen to use loan-to-value, debt-to-income, and borrower
credit score as predictors. Indeed, much research indicates that
these features are highly predictive. [Cooper (2018)][1] presents 
a section detailing this. In addition, training a Random Forest 
model for feature subset selection coincides with these results.
For details, see my repository, [here][2].

For all models, class imbalance is a concern. As demonstrated
below, most accounts have not been referred to foreclosure.

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/fc_stat_freq.png)

To address the issue, we implement over & under sampling. Testing
indicated that under-sampling was more successful; however, the
sampling ratio varied. In addition, it is important to measure
model success on metrics such as precision and recall, instead of
model accuracy. With such a class imbalance, any model can reach
over 90% accuracy by simply predicting non-default for all 
training instances.

### Training Data

### K-Nearest-Neighbors (KNN)

For the KNN model, we tested models with k in the range 3, 20. 

In addition, we tested sampling ratios between .1 and 1. Testing indicates that
the equilibrium is found between recall for the classes at .8. 

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/knn_samling_ratio_recall.png)

### Deep Learning

### Random Forest

## Lessons Learned
* Appify, etc.


## Resources
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_File_layout.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/performance-sample-file.txt
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_Glossary.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_FAQs.pdf
* http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html

[1]: https://www.researchgate.net/publication/330303425_A_Deep_Learning_Prediction_Model_for_Mortgage_Default_A_Deep_Learning_Prediction_Model_for_Mortgage_Default "Cooper, Michael. (2018). A Deep Learning Prediction Model for Mortgage Default A Deep Learning Prediction Model for Mortgage Default. 10.13140/RG.2.2.21506.12487."
[2]: https://github.com/dgillis91/fnma_loan_performance "Random Forest Model over FNMA Mortgage Default Data"