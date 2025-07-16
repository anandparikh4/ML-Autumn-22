import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder

def encode(df):
    # use in-built label encoder to convert the classifications "Abnormal" and "Normal" to 0 and 1, respectively
    patient_encoder = LabelEncoder()
    patient_encoder.fit(df['Class_att'])                                # pass in the name of Column to encode
    df['Class_att'] = patient_encoder.transform(df['Class_att'])        # transform and assign 0,1 values
    return df

def extract(df):
    # splitting data into attributes and target value

    X = df.drop([df.columns[-1]], axis=1)       # remove the last column
    y = df[df.columns[-1]]                      # take only the last column

    return X, y

def train_test_split(X,y,frac):
    n_train = math.floor(frac * X.shape[0])          # train -> fraction * total
    n_test = math.floor((1 - frac) * X.shape[0])     # test -> (1-fraction) * total

    X_train = X[:n_train]           # slice all arrays accordingly to create Training and Testing data
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test =  y[n_train:]

    return X_train,X_test,y_train,y_test

def remove_outliers(df,a,b):
    # remove outliers that have values more than (2 * mean + 5 * std)
    col_names = list(df.columns)
    means = {}                          # storing means and standard deviations of each column
    standard_deviations = {}

    ncols = len(col_names)
    ncols = ncols - 1                   # to ignore the classification column (label value)

    for col in col_names[:-1]:
        mean = df[col].mean()               # calculate mean and std
        standard_deviation = df[col].std()
        means[col] = mean
        standard_deviations[col] = standard_deviation

    outliers = [0] * len(df)             # calculate the number of outliers in each row and store in this array

    for col in col_names[:-1]:
        values = df[col].tolist()        # temporarily convert to a list for easy edit and access
        row = 0
        for v in values:
            if((v > a * means[col] + b * standard_deviations[col])):        # if value > 2*mean + 5*std, increment outlier value for that row
                outliers[row] = outliers[row] + 1
            row = row+1

    row = 0
    x = 0               # count the number of rows having outliers

    print("The following row numbers have outlying features:")
    while(row < len(df)):
        if(outliers[row] > 0):    # these are the rows containing outlying data, so these rows have been discarded, since they can severely misguide our model
            print(df.loc[[row]])
            #print("\n")
            df = df.drop(df.index[row])
            x = x+1
        row = row+1

    return df

def calculate_priors(y_train):
    # calculate the prior probabilities of NORMAL and ABNORMAL classification
    priors = []
    priors.append(len(y_train[y_train == 0]) / len(y_train))        # simply take the fractions of 0 and 1 out of total samples
    priors.append(len(y_train[y_train == 1]) / len(y_train))
    return priors

def calculate_likelihoods(X_train,y_train,i,attr_val,classification):
    # basically, take only those examples which have a GIVEN classification (abnormal or normal), and forget others
    X_train = X_train[y_train == classification]
    X_train = X_train.T     # temporarily transpose to find mean and std easily

    mean = np.mean(X_train[i])
    std = np.std(X_train[i])

    # now, find the probability that attr_name has value attr_val GIVEN a fixed classification
    prob = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((attr_val-mean)**2 / (2 * std**2 )))
    # use the pdf of normal distribution to achieve this
    return prob

def Naive_Bayes_Classifier(X_train,X_test,y_train,a=0,laplace = False):
    # find priors of both classifications and store in array priors
    priors = calculate_priors(y_train)

    y_predicted = []        # to store the predicted classifications of X_test

    for x in X_test:
        likelihood = [1] * 2

        # for the Naive Bayes Classifier, we are assuming independence of variables
        # so, likelihoods are the product of individual probabilities
        for i in range(X_train.shape[1]):
            if(laplace):
                likelihood[0] *= (calculate_likelihoods(X_train,y_train,i,x[i],0)+a)/(1 + a)       # add a fraction "a" to the numerator and denominator of all the probabilities while calculating likelihoods
                likelihood[1] *= (calculate_likelihoods(X_train,y_train,i,x[i],1)+a)/(1 + a)
            else:
                likelihood[0] *= calculate_likelihoods(X_train,y_train,i,x[i],0)      # The laplace correction is an option provided to the user
                likelihood[1] *= calculate_likelihoods(X_train,y_train,i,x[i],1)

        posteriors = [0] * 2
        posteriors[0] = likelihood[0] * priors[0]           # now, posterior probability = prior probability * likelihood
        posteriors[1] = likelihood[1] * priors[1]

        y_predicted.append(np.argmax(posteriors))           # final classification is the one with higher posterior probability

    return y_predicted

def accuracy(y_predicted,y_test):
    # simply find percentage of instances correctly classified
    cnt = 0
    for i in range(len(y_test)):
        if(y_predicted[i] == y_test[i]):
            cnt += 1
    return cnt / len(y_test)

def cross_validation(X,y):
    # Doing a 5-fold cross validation, where the training set is split into 5 parts.
    # Each split becomes the testing part once, with the remaining 4 concatenated to build the training part.
    # Basically a 80-20 split of the 70% training data, and each data point (and each split) is used 1 time in testing, 4 times in training
    foldX = np.array_split(X,5)
    foldy = np.array_split(y,5)

    X_train = [0] * 5
    X_test = [0] * 5
    y_train = [0] * 5
    y_test = [0] * 5
    y_predicted = [0] * 5
    acc = [0] * 5

    X_train[0] = np.concatenate((foldX[0],foldX[1],foldX[2],foldX[3]))
    X_test[0] = foldX[4]
    y_train[0] = np.concatenate((foldy[0],foldy[1],foldy[2],foldy[3]))
    y_test[0] = foldy[4]
    y_predicted[0] = Naive_Bayes_Classifier(X_train[0] , X_test[0] , y_train[0])
    acc[0] = accuracy(y_predicted[0] , y_test[0])

    X_train[1] = np.concatenate((foldX[0],foldX[1],foldX[2],foldX[4]))
    X_test[1] = foldX[3]
    y_train[1] = np.concatenate((foldy[0],foldy[1],foldy[2],foldy[4]))
    y_test[1] = foldy[3]
    y_predicted[1] = Naive_Bayes_Classifier(X_train[1] , X_test[1] , y_train[1])
    acc[1] = accuracy(y_predicted[1] , y_test[1])

    X_train[2] = np.concatenate((foldX[0],foldX[1],foldX[3],foldX[4]))
    X_test[2] = foldX[2]
    y_train[2] = np.concatenate((foldy[0],foldy[1],foldy[3],foldy[4]))
    y_test[2] = foldy[2]
    y_predicted[2] = Naive_Bayes_Classifier(X_train[2] , X_test[2] , y_train[2])
    acc[2] = accuracy(y_predicted[2] , y_test[2])

    X_train[3] = np.concatenate((foldX[0],foldX[2],foldX[3],foldX[4]))
    X_test[3] = foldX[1]
    y_train[3] = np.concatenate((foldy[0],foldy[2],foldy[3],foldy[4]))
    y_test[3] = foldy[1]
    y_predicted[3] = Naive_Bayes_Classifier(X_train[3] , X_test[3] , y_train[3])
    acc[3] = accuracy(y_predicted[3] , y_test[3])

    X_train[4] = np.concatenate((foldX[1],foldX[2],foldX[3],foldX[4]))
    X_test[4] = foldX[0]
    y_train[4] = np.concatenate((foldy[1],foldy[2],foldy[3],foldy[4]))
    y_test[4] = foldy[0]
    y_predicted[4] = Naive_Bayes_Classifier(X_train[4] , X_test[4] , y_train[4])
    acc[4] = accuracy(y_predicted[4] , y_test[4])

    index = np.argmax(acc)      # get the index of training and testing slpits which gave the highest accuracy, and use them to train the model again

    print("Accuracy2 (Average accuracy of cross-validation (within the training set itself))")
    print(np.mean(acc))

    #print(acc)
    #print(index)

    return X_train[index] , y_train[index]      # return those training and testing sets

if __name__ == "__main__":

    # patients' dataset
    print("\nPatients' Dataset")

    # Read dataset
    df = pd.read_csv("Train_D_Bayesian.csv")

    # Encode the non-numeric label values to numeric values
    df = encode(df)

    # Remove outlying data, according to definition provided in the problem statement
    df = remove_outliers(df,2,5)

    # First shuffle randomly, since shuffling the lists after splitting is incorrect
    df = df.sample(frac=1)

    # Extract attributes and target values from data
    X , y = extract(df)

    # Spilt into training and testing data (70 - 30 split here, so training fraction is 0.7)
    X_train, X_test, y_train, y_test = train_test_split(X,y,0.7)

    # Converting all to numpy since it is easy and fast
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Find the predictions and accuracy by directly training on the 70% training data
    y_predicted = Naive_Bayes_Classifier(X_train,X_test,y_train)

    acc = accuracy(y_predicted,y_test)

    print("Accuracy1 (simple case, 70% training, 30% testing)")
    print(acc)         # directly running on 70-30 split

    final_X_train , final_y_train = cross_validation(X_train,y_train)       # do a 5-fold cross validation and take the best training and testing instance

    final_y_predicted = Naive_Bayes_Classifier(final_X_train,X_test,final_y_train)      # train the model again on cross-validated training set

    final_acc = accuracy(final_y_predicted , y_test)

    print("Accuracy3 (best split of cross validation as training, 30% testing)")
    print(final_acc)                # print the final accuracy achieved after cross-validation

    # Now, train the model with Laplace Correction:

    y_predicted_with_laplace = Naive_Bayes_Classifier(X_train,X_test,y_train,0.09,True)

    acc_with_laplace = accuracy(y_predicted_with_laplace,y_test)

    print("Accuracy4 (70% training, 30% testing, with Laplace correction)")
    print(acc_with_laplace)



