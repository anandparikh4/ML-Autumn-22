# ML Assignment 2
# Autumn 2022
#
# Authors :-
#     Anand Manojkumar Parikh (20CS10007)
#     Soni Aditya Bharatbhai (20CS10060)
#
# Supervised Learning Methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from collections import Counter

def encode(df):
    # Encode categorical data as follows
    df.loc[df['sex'] == 'M','sex'] = 0      # Males -> 0
    df.loc[df['sex'] == 'F','sex'] = 1      # Females -> 1
    df.loc[df['sex'] == 'I','sex'] = 2      # Infants -> 2

    return df

def extract(df):
    # splitting data into attributes and target value (the target value is the last column)
    X = df.drop([df.columns[-1]], axis=1)       # remove the last column
    y = df[df.columns[-1]]                      # take only the last column

    return X, y

def train_test_split(X,y,frac):
    # Split available data into training and testing sets as per the given fraction
    n_train = math.floor(frac * X.shape[0])          # train -> fraction * total
    n_test = math.floor((1 - frac) * X.shape[0])     # test -> (1-fraction) * total

    X_train = X[:n_train]           # slice all arrays accordingly to create Training and Testing data
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train,X_test,y_train,y_test

def normalize(df):
    # Standard scaler normalization
    for col in df.columns:
        if col != 'rings':                             # Normalize all attributes except the target value
            df[col] = (df[col] - df[col].mean()) / df[col].std()        # val  = (val - mean) / standard_deviation
    return df

def accuracy(y_pred,y_test):
    # simply find percentage of instances correctly classified
    return len(y_pred[y_pred == y_test])/len(y_test)

def backward_elimination(X_train,X_test,y_train,y_test,model):
    # All features present initially, then removed sequentially by a greedy approach
    features = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight"]
    model.fit(X_train,y_train)                        # fit to training data
    y_pred = model.predict(X_test)                    # find classified predictions
    prev_acc = accuracy(y_pred,y_test)        # calculate initial accuracy

    print("Backward Elimination begins:")
    print("Initial accuracy")
    print(prev_acc)

    while True:             # run an infinite while loop that removes features step-wise and breaks when max accuracy < prev accuracy
        max_acc = 0         # initialize max accuracy to zero
        to_remove = None    # nothing to be removed yet

        for feature in features:                            # iterate over all existing features
            X_train_2 = X_train.drop([feature], axis=1)     # drop this feature temporarily
            X_test_2 = X_test.drop([feature],axis=1)        # from both train and test sets

            model.fit(X_train_2,y_train)                    # then re-fit the model
            y_pred = model.predict(X_test_2)                # predict classifications
            acc = accuracy(y_pred,y_test)                   # measure accuracy

            if(acc > max_acc):          # if this accuracy is larger than max accuracy
                max_acc = acc           # update max accuracy
                to_remove = feature     # and remember which feature to drop

        print("Max accuracy achieved on removing feature : " + to_remove)
        print(max_acc)
        print(X_train.columns)

        if(max_acc < prev_acc):         # if maximum accuracy on removing any existing feature is smaller than previous accuracy, terminate process
            print("Feature : " + to_remove + " not removed")
            break

        else:                           # else, re-iterate after removing this feature permanently
            X_train = X_train.drop([to_remove], axis=1)
            X_test = X_test.drop([to_remove], axis=1)
            features.remove(to_remove)  # also delete from list of existing features, so in next the iteration, this feature is not explored again
            prev_acc = max_acc          # update prev_acc
            print("Feature removed : " + to_remove)

    print("Backward Elimination ends")
    return features         # return the final subset of features that survive

def ensemble(svm_quad,svm_rbf,mlp,X_train,X_test,y_train,y_test):
    svm_quad.fit(X_train,y_train)       # fit all 3 models to the training data
    svm_rbf.fit(X_train,y_train)
    mlp.fit(X_train,y_train)

    y_pred = []

    y_pred.append(svm_quad.predict(X_test))         # make a list of lists to store the predictions of all 3 models
    y_pred.append(svm_rbf.predict(X_test))
    y_pred.append(mlp.predict(X_test))

    # use collections.Counter to implement the max-voting technique
    y_pred_max_vote = [Counter(column).most_common(1)[0][0] for column in zip(*y_pred)]     # in each column, find the most frequent of the 3 values and make a list of those
    return y_pred_max_vote      # return the max-voted list

if __name__ == '__main__':

    # Abalone dataset
    print("\nAbalone Dataset")
    column_names = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
    df = pd.read_csv("abalone.data",names=column_names)

    # The .names clearly specifies that there is no missing data, so no need to handle that

    # Encode the non-numeric label values to numeric values
    df = encode(df)

    # Normalize
    df = normalize(df)

    # First shuffle randomly, since shuffling the lists after splitting is incorrect
    df = df.sample(frac=1)

    # Extract attributes and target values from data
    X, y = extract(df)

    # Spilt into training and testing data (80 - 20 split here, so training fraction is 0.8)
    X_train, X_test, y_train, y_test = train_test_split(X,y,0.8)

    # Support Vector Classification (SVC)

    # Make objects of SVC class with required kernels
    sv_linear_classifier = SVC(kernel='linear')                  # linear
    sv_quadratic_classifier = SVC(kernel='poly',degree=2)        # quadratic, so polynomial with degree 2
    sv_rbf_classifier = SVC(kernel='rbf')                        # radial basis function

    # Fit the models onto the training data
    sv_linear_classifier.fit(X_train,y_train)
    sv_quadratic_classifier.fit(X_train,y_train)
    sv_rbf_classifier.fit(X_train,y_train)

    # Predict the classified values of the test data
    y_pred_linear = sv_linear_classifier.predict(X_test)
    y_pred_quadratic = sv_quadratic_classifier.predict(X_test)
    y_pred_rbf = sv_rbf_classifier.predict(X_test)

    # Find accuracy score of the output values
    svr_linear_accuracy = accuracy(y_pred_linear,y_test)
    svr_quadratic_accuracy = accuracy(y_pred_quadratic,y_test)
    svr_rbf_accuracy = accuracy(y_pred_rbf,y_test)

    print("Linear SVC accuracy score:")
    print(svr_linear_accuracy)
    print("Quadratic SVC accuracy score:")
    print(svr_quadratic_accuracy)
    print("Radial Basis Function SVC accuracy:")
    print(svr_rbf_accuracy)

    # Multi-layer Perceptron Classification (MLP)

    # Make objects of MLP Classifier class using a SGD (Stochastic Gradient Descent) solver, having learning rate 0.001 and batch size 32
    mlp_1_layer = MLPClassifier(hidden_layer_sizes=(16), solver='sgd',learning_rate='constant',learning_rate_init=0.001,batch_size=32)           # 1 hidden layer with 16 nodes
    mlp_2_layers = MLPClassifier(hidden_layer_sizes=(256,16), solver='sgd',learning_rate='constant',learning_rate_init=0.001,batch_size=32)      # 2 hidden layers with 256 and 16 nodes, respectively

    # Fit the training models onto the training data
    mlp_1_layer.fit(X_train,y_train)
    mlp_2_layers.fit(X_train,y_train)

    # Predict the classified values of the test data
    y_pred_1_layer = mlp_1_layer.predict(X_test)
    y_pred_2_layers = mlp_2_layers.predict(X_test)

    # Find accuracy score of the output values
    mlp_1_layer_accuracy = accuracy(y_pred_1_layer,y_test)
    mlp_2_layers_accuracy = accuracy(y_pred_2_layers,y_test)

    print("1-layer MLP accuracy score:")
    print(mlp_1_layer_accuracy)
    print("2-layer MLP accuracy score:")
    print(mlp_2_layers_accuracy)

    # Find which MLP model works better (the one higher accuracy score)
    mlp_best = mlp_1_layer                                  # assume mlp_1 is better
    if(mlp_2_layers_accuracy > mlp_1_layer_accuracy):       # then check if mlp_2 gives higher accuracy
        mlp_best = mlp_2_layers                             # if it does, replace with mlp_2

    # Vary learning rate on the model mlp_best
    learning_rate = [0.1,0.01,0.001,0.0001,0.00001]     # various learning rates
    accuracies = []                                     # accuracy scores corresponding to each learning rate

    for i in range(0,5):                                    # For each learning rate
        mlp_best.learning_rate_init = learning_rate[i]      # Set the learning rate
        mlp_best.fit(X_train,y_train)                       # Fit again on training data (since field values have changed)
        y_pred_mlp_best = mlp_best.predict(X_test)          # Predict target values
        acc = accuracy(y_pred_mlp_best,y_test)              # Find accuracy score
        accuracies.append(acc)                              # Store it

    mlp_best.learning_rate_init = 0.001     # Don't forget to restore the object fields back to their original values after the FOR loop above
    mlp_best.fit(X_train,y_train)           # and re-fit to the training examples

    print(accuracies)

    # Plotting the Learning Rate V/S Accuracy Graph

    plt.plot(learning_rate,accuracies,'bo',linestyle='--')
    plt.xlabel("X-Axis -> Learning Rate")
    plt.ylabel("Y-Axis -> Accuracies")
    plt.title("MLP Graph")
    for xy in zip(learning_rate, accuracies):
        plt.annotate('(%.6f, %.6f)' % xy, xy=xy)
    plt.show()

    # Backward Elimination (self-implemented)

    mlp_best_copy = mlp_best        # make copies of all the parameter to pass, since they will be modified in the function
    X_train_copy = X_train
    X_test_copy = X_test
    y_train_copy = y_train
    y_test_copy = y_test
    best_features = backward_elimination(X_train_copy,X_test_copy,y_train_copy,y_test_copy,mlp_best_copy)       # call backward_elimination function (implemented above)
    print("Best features selected by backward elimination are : ")
    print(best_features)            # This is the best feature subset found using backward elimination

    # Ensemble Learning (self-implemented)

    y_pred_max_vote = ensemble(sv_quadratic_classifier,sv_rbf_classifier,mlp_best,X_train,X_test,y_train,y_test)    # call ensemble function (implemented above)
    ensemble_max_vote_accuracy = accuracy(np.array(y_pred_max_vote),y_test.to_numpy())         # This is the accuracy found out by using a max-vote technique amongst 3 models : Quadratic SVC, RBF SVC and the best of mlp_1_layer and mlp_2_layers
    print("Ensemble Learning accuracy score (using max-vote technique) :")
    print(ensemble_max_vote_accuracy)
