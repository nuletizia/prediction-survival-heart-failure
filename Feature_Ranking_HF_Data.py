import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu, pearsonr, chisquare
from sklearn.feature_selection import chi2,SelectKBest,SelectFromModel

import matplotlib.pyplot as plt

# read heart failure dataset CSV file
dataset = pd.read_csv('HFCD.csv', delimiter=',')

ranking_method = 'mannwhitneyu'
# prepare all data
X = dataset.iloc[:, :-1]
X_no_time = X.iloc[:,:-1]
y = dataset.iloc[:, -1]
X_no_time.info()

r,c = X_no_time.shape

# select the death events and check the features related
y_D = y[y==1]
X_D = X[y==1]
death_events,_ = X_D.shape

# select the survival events
y_S = y[y==0]
X_S = X[y==0]

if ranking_method == 'mannwhitneyu':
    # for every feature do the Whitney test and rank them based on the p value
    U = np.zeros((1,c))
    p = np.zeros((1,c))
    ranks = np.zeros((1,c))

    for i in range(c):
        U[0,i], p[0,i] = mannwhitneyu(X.iloc[:,i], X_D.iloc[:,i])

    sorted_indices = np.argsort(p)

    for i in range(c):
        ranks[0,i] = np.where(sorted_indices[0,:]==i)[0][0]

    # 1st row describes the rank of each column
    # 2nd row provides the corresponding p value
    data_for_frame = np.concatenate((ranks.astype('int8'),p),axis=0)

    # pass column names in the columns parameter
    df_mann = pd.DataFrame(data_for_frame, columns = X_no_time.columns)
    print(df_mann)

    # plot feature importance
    plt.bar(X_no_time.columns[sorted_indices[0,:]].values, p[0,sorted_indices[0,:]])
    plt.xlabel('Feature rank')
    plt.ylabel('p value')
    plt.title('Feature ranking using mannwhitneyu')
    plt.show()

elif ranking_method == 'pearson':
    # for every feature compute the correlation and rank them based on the p value
    R = np.zeros((1,c))
    p = np.zeros((1,c))
    ranks = np.zeros((1,c))
    for i in range(c):
        R[0,i], p[0,i] = pearsonr(X.iloc[np.random.choice(range(r), death_events, replace=False),i], X_D.iloc[:,i])
    # does not work perfectly...
    sorted_indices = np.argsort(1-abs(R))

    for i in range(c):
        ranks[0,i] = np.where(sorted_indices[0,:]==i)[0][0]

    # 1st row describes the rank of each column
    # 2nd row provides the corresponding R value
    data_for_frame = np.concatenate((ranks.astype('int8'),R),axis=0)

    # pass column names in the columns parameter
    df_corr = pd.DataFrame(data_for_frame, columns = X_no_time.columns)
    print(df_corr)

elif ranking_method == 'chisquare':
    # select k best features based on chi2 method
    num_features = 4
    chi2_selector = SelectKBest(chi2, k=num_features)
    X_kbest = chi2_selector.fit_transform(X_no_time, y)
    indices_X_kbest = chi2_selector.get_support()
    print(f'Indices of the best {num_features} features are: {X_no_time.columns[indices_X_kbest].values}')

elif ranking_method == 'randomforest':
    # select the best features based on random forest method
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_no_time, y)
    # Print the names of the most important features
    for feature_list_index in sel.get_support(indices=True):
        print(X_no_time.columns[feature_list_index])

elif ranking_method == 'pca':
    # identify the number of features needed using PCA, no labels (unsupervised)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_no_time)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    plt.plot(range(c), explained_variance, lw=2)
    plt.xlabel('PCA feature component')
    plt.ylabel('Variance')
    plt.show()
