import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import argparse

class BinaryClassifierHF():
    def __init__(self, prediction_method, trials = 10, do_reduction = False):
        self.prediction_method = prediction_method
        self.trials = trials
        self.do_reduction = do_reduction

    def load_data(self):
        # read heart failure dataset CSV file
        dataset = pd.read_csv('HFCD.csv', delimiter=',')

        dataset.info()

        # select only 2 columns based on the results of the feature selection code
        if do_reduction:
            best_features = [4,7] # ejection fraction and serum creatinine
            X = dataset.iloc[:,best_features]
            print(X.columns)
        else:
            X = dataset.iloc[:,:-1]
        y = dataset.iloc[:, -1]
        return X,y

    def classify_heart_data(self,X,y):
        # create classifier based on the prediction method

        score_vector = np.zeros((1,self.trials))
        MCC_vector = np.zeros((1,self.trials))
        f1_vector = np.zeros((1,self.trials))

        if self.prediction_method == 'naive_bayes':
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)
                # train a naive bayes classifier
                NB = GaussianNB()
                NB.fit(X_train, y_train)

                # testing phase
                y_true, y_pred = y_test, NB.predict(X_test)
                # get the score
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del NB

        elif self.prediction_method == 'decision_tree':
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)
                # train a decision tree classifier
                DecisionTree = DecisionTreeClassifier()
                DecisionTree.fit(X_train, y_train)
                DecisionTree.score(X_train, y_train)

                # testing phase
                y_true, y_pred = y_test, DecisionTree.predict(X_test)
                # get the score
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del DecisionTree

        elif self.prediction_method == 'SVM':
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)
                # train a SVM classifier
                svm = SVC(kernel='rbf')
                svm.fit(X_train, y_train)

                # testing phase
                y_true, y_pred = y_test, svm.predict(X_test)
                # get the scores
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del svm

        elif self.prediction_method == 'random_forest':
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)
                # train a random forest classifier
                RF = RandomForestClassifier(max_depth=2, random_state=random_state)
                RF.fit(X_train, y_train)

                # testing phase
                y_true, y_pred = y_test, RF.predict(X_test)
                # get the scores
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del RF

        elif self.prediction_method == 'gradient_boosting':
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)
                # train gradient boosting classifier
                GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=random_state)
                GB.fit(X_train, y_train)

                # testing phase
                y_true, y_pred = y_test, GB.predict(X_test)
                # get the scores
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del GB

        elif self.prediction_method == 'logistic_regression':
            # Logistic regression requires normalized data, so we applied a MinMax scaler between 0 and 1
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)

                min_max_scaler = MinMaxScaler()
                X_train_minmax = min_max_scaler.fit_transform(X_train)

                # train logistic regression
                LogReg = LogisticRegression(random_state=random_state)
                LogReg.fit(X_train_minmax, y_train)

                # testing phase
                y_true, y_pred = y_test, LogReg.predict(min_max_scaler.fit_transform(X_test))
                # get the scores
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del LogReg

        elif self.prediction_method == 'neural_network':
            # Neural network classification may require normalized data, so we applied a MinMax scaler between 0 and 1
            for i in range(self.trials):
                random_state = np.random.RandomState(i)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state = random_state)

                min_max_scaler = MinMaxScaler()
                X_train_minmax = min_max_scaler.fit_transform(X_train)

                # train a NN classifier
                MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(75,), random_state=random_state, early_stopping =True, solver='adam')

                MLP.fit(X_train_minmax, y_train)

                # testing phase
                y_true, y_pred = y_test, MLP.predict(min_max_scaler.fit_transform(X_test))
                # get the scores
                score_vector[0,i] = accuracy_score(y_true, y_pred)
                MCC_vector[0,i] = matthews_corrcoef(y_true, y_pred)
                f1_vector[0,i] = f1_score(y_true, y_pred)
                del MLP
        return np.mean(score_vector),np.mean(MCC_vector),np.mean(f1_vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_method', help='Prediction method to use to train the classifier. Choose among: naive_bayes, decision_tree, random_forest, SVM, gradient_boosting, logistic_regression or neural_network.', default='random_forest')
    parser.add_argument('--trials', help='Number of MC simulations', default=10)
    parser.add_argument('--do_reduction', help='Train classifier using only the best two features', default=False)

    args = parser.parse_args()

    prediction_method = args.prediction_method
    trials = int(args.trials)
    do_reduction = bool(args.do_reduction)

    BCHF = BinaryClassifierHF(prediction_method,trials,do_reduction)
    X,y = BCHF.load_data()
    acc,MCC,f1 = BCHF.classify_heart_data(X,y)

    print(f'The accuracy, MCC and f1 scores of the {prediction_method} method are: {acc}, {MCC} and {f1}. Is the method using only the best 2 features? {do_reduction}')
