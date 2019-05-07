# mortgage_default_model

## Introduction

This is a project to compare various models for predicting serious 
mortgage default. The entry level model is a multivariate logistic 
regression model. In addition, we implemented a KNN model, and a 
more complex neural network. We find that, with the data used for
the task, more advanced neural network topologies lead to better
performance in prediction. 

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

We've chosen to use loan-to-value, debt-to-income, age, and 
borrower credit score as predictors. Indeed, much research 
indicates that these features are highly predictive. 
[Cooper (2018)][1] presents a section detailing this. In 
addition, training a Random Forest model for feature subset 
selection coincides with these results. For details, see my 
repository, [here][2].

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
This was the place where I likely learned the most. There are
a few things that need to be done to keep the training & testing
data independent. 

First, when scaling your data, it should be
done independently on the training & testing data. In this case, 
we used min-max-scaling. 

Next, I chose to under sample after splitting the training/testing
data. This results in some wasted data; however, it allows you to
test on a sample with a distribution similar to the population. 

### K-Nearest-Neighbors (KNN)

For the KNN model, we tested models with k in the range 3, 20. 

In addition, we tested sampling ratios between .1 and 1. Testing 
indicates that the equilibrium is found between recall for the 
classes at .8. 

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/knn_samling_ratio_recall.png)

At this ratio, we have precision of 20%, and recall of 68% for 
the default class. For non-default, the results are 95.95%, and
70%, respectively. 

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/knn_precision_recall_table_sample_rate.png)

Testing with a sampling ratio of 0.8 indicates that 19 is a good
selection of k. 

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/knn_k_selection.png)

### Deep Learning

We proceed to show that a deep learning model leads to 
increased performance over the classic logistic regression
model.

The logistic regression model is created using a single
sigmoid perceptron. The model we settled on for deep 
learning consists of two hidden layers, each with 
64 nodes, and a single sigmoid output layer.

For the complex model, we to see that increasing the
sample ratio decreases the recall for the majority
class, and increases the recall for the minority class. 
Interestingly, increasing the sampling ratio decrease the 
precision for the minority class, and increase it for the
majority.

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/complex_model_prec_recall_p1.png)
*ratio = .1*

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/complex_model_prec_recall_p5.png)
*ratio = .5*

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/complex_model_prec_recall_p7.png)
*ratio = .7*

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/complex_model_prec_recall_p10.png)
*ratio = 1*

For the simple, logistic regression model, we see similar
results for the sampling ratio. However, the complex model
performs substantially better as the dataset becomes more
imbalanced. 

For comparison, we have the results of the log model,
followed by the more complex topology with a sampling
rate of .1.

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/log_model_prec_recall_p1.png)
![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/complex_model_prec_recall_p1.png)

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/log_model_prec_recall_p5.png)

*ratio = .5*

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/log_model_prec_recall_p7.png)

*ratio = .7*

![Alt](https://raw.githubusercontent.com/dgillis91/mortgage_default_model/master/analysis/log_model_prec_recall_p10.png)

*ratio = 1*

In addition, we see that the complex model has better
precision & recall scores in most ranges. 

## Lessons Learned
* It is best to keep your configurations in a file. I'm
fond of the JSON format for ease of use, though it has some
drawbacks. 
* When you run a model on a configuration file, output the
configurations and the results to a unique file. This
allows you to compare the results across topologies and 
hyperparameters. You may also choose to write your results
to a simple database. sqlite has a simple python api, though
it is single user.
* Keep a history of actions you take in tuning models. It is
very difficult to go back and retrace your steps. 
* Use pipelines where possible. They make transformations much 
more legible. For an example, we trained an arbitrarily 
complicated model with some additional features. The
transforms for this model can be viewed in model_pipeline.py.
The model itself is in newmodel.py. For a detailed analysis,
look into sklearn.pipeline.


## Resources
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_File_layout.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/performance-sample-file.txt
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_Glossary.pdf
* https://loanperformancedata.fanniemae.com/lppub-docs/FNMA_SF_Loan_Performance_FAQs.pdf
* http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html

[1]: https://www.researchgate.net/publication/330303425_A_Deep_Learning_Prediction_Model_for_Mortgage_Default_A_Deep_Learning_Prediction_Model_for_Mortgage_Default "Cooper, Michael. (2018). A Deep Learning Prediction Model for Mortgage Default A Deep Learning Prediction Model for Mortgage Default. 10.13140/RG.2.2.21506.12487."
[2]: https://github.com/dgillis91/fnma_loan_performance "Random Forest Model over FNMA Mortgage Default Data"