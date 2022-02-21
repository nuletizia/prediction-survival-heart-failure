# Survival Prediction of Patients with Heart Failure
Simple binary classification of electronic health records of patients with cardiovascular heart diseases


This repository contains an unofficial and preliminary implementation of the methods described in: 
"Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone"
https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5

The work was done as a simple student project for the course of "Data Mining" at the University of Klagenfurt. 

The repository is divided into two components: 
1. Survival prediction using different binary classifiers on the entire or reduced dataset
2. Feature ranking using some of the methods described in the original paper

<img src="https://github.com/nuletizia/prediction-survival-heart-failure/blob/main/feature_ranking_mwu.png" width=700>

# Survival Prediction

Run the following command to get accuracy, MCC and F1 score exploiting the full dataset and the random forest classifier using 10 Monte Carlo simulations:
> python Prediction_HF_Data.py

If you want to change the prediction method (naive_bayes, decision_tree, random_forest, SVM, gradient_boosting, logistic_regression or neural_network), run the following command:

> python Prediction_HF_Data.py --prediction_method random_forest

If you want to increase to 100 the number of trials/simulations (different seeds), use:

> python Prediction_HF_Data.py --trials 100

If you want to build a classifier using only the best 2 features found via the feature selection/ranking code "Feature_Ranking_HF_Data.py"", put the do_reduction flag to True as follows:

> python Prediction_HF_Data.py --do_reduction True

For instance, if you want to use the logistic regression with 100 different independent realizations using only the 2 best features of the dataset, run:

> python Prediction_HF_Data.py --prediction_method logistic_regression --trials 100 --do_reduction True

# Feature Ranking

Run the following command to get the feature ranking using the Mann-Whitney-U test

> python Feature_Ranking_HF_Data.py

Pearson, Chi-square and random forest ranking methods are also tentatively implemented. Change the ranking_method variable to try one of them.
PCA analysis is available on the normalized data.
