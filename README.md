# Cajamar Predictive Modelling Challenge (PyConEs16)

Solution for the Cajamar predictive modelling datathon (PyConEs16) - http://www.cajamardatalab.com/datathon-cajamar-pythonhack-2016/

I tried several models and classifiers, getting the best results with a combination of a feature selection by an extraTrees, and a kNN classifier.

The biggest problem was the extremely imbalanced training dataset: only 1.6% of the instances of the positive class. Due to this, a SMOTE oversampling was applied to balance the training dataset (Library https://github.com/scikit-learn-contrib/imbalanced-learn).

I tried lots of classifiers and methods (xgboost, randomForest...) but due to the limited time given, I couldn't do parameter optimizations and had to stick with the classifier applied in the code, which was far from being perfect, but at least classified correctly more than half of the positive classes in the test splits.

Although I didnt get into the top 3, I learnt a lot!

The winner solution can be found here: https://github.com/masdeseiscaracteres/pythonhack2016

## The datasets
The datasets can not be published, due to the terms and conditions of the contest. 
