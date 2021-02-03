# estimators_demo

This is a demo of several estimators. The main module runs a list of them, the winner
will be selected and fed the entire dataset to fit the data well.
Then the validation set can be used to predict unseen output data.
This data is saved under validation_prediction.csv.
A plot to each model is exported to plot.pdf.

### Start with:
Python main.py training.xml validation.xml

training.xml n-dimensional data set for training of models (labeled).

validation.xml 1-dimensional target values (co-domain) for just export (unlabeled).

As further parameter, set OPTIMIZE_MODELS = True to perform full parameter search each call of main.py 

### Export files are the following

Visualization:
https://github.com/MC-program/estimators_demo/blob/master/plot.pdf

CAUTION: Even each page get a single plot, there are flaws on some pages! 

But I got 100s warnings from matplotlib.font_manager (findfont()) on my system. You might be able to generate it without problems on another system.

https://github.com/MC-program/estimators_demo/blob/master/validation_prediction.csv

### Comments 
There are 6 different models, see estimators.py: 

* K-nn, 
* SVM, 
* linear model, 
* neural network, 
* gradient boosting ensemble 
* decision tree model.

This is a pick for very small datasets (200 observations). For bigger sizes, sparse data, or high-dimensional data, this set would be at least extended. For example: Feature selection and compressing. On the neural network side, there can be way more fancy models (deep learning). Etc.

Modules of the demo are:

* plot for plotting into PDF's

* debug_package for debug and warnings

* xml_parser for jobs related to files

* estimators with the machine learning models in SciKit-learn

* main Start and pass (file) arguments


*main* reads in all data first, and min-max-normalizes the input data. Then parameter search for each model is executed, and the best model is kept. This is trained onto the full labeled input data. Then the the prediction is perfomed onto the unlabeled data and the results come into the export files.

The *winner* model with the small pre-defined dataset is k-nn (k=5) !


### TODO

Warnings from matplotlib.font_manager and some flaws in the plot pdf export.

