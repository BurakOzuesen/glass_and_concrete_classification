import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

glass_df = pd.read_csv("glass.csv")
concrete_df = pd.read_csv("Concrete_Data_Yeh.csv")
concrete_df["age"] = concrete_df["age"].astype(float)


def normalization(df): # min-max normalization
    normalized_df = df.copy(deep=True)  # create a new dataFrame independent from the original one.
    for column_name in normalized_df.columns:
        
        if column_name != "Type":
            column_max = normalized_df[column_name].max()
            column_min = normalized_df[column_name].min()
            for i in range(len(normalized_df[column_name])):
                normalized_df[column_name][i] = (normalized_df[column_name][i] - column_min) / (column_max - column_min)
            
    return normalized_df


normalized_glass_df = normalization(glass_df)
normalized_concrete_df = normalization(concrete_df)

def cross_validation(df): # k-fold cross validation with k=5

    indices = np.arange(df.shape[0])
    np.random.shuffle(indices)

    bucket_1 = []
    bucket_2 = []
    bucket_3 = []
    bucket_4 = []
    bucket_5 = []

    for index, item in enumerate(indices):          # 5erli sub arrayler olarak da ele alınabilir ama galiba en hızlısı bu
        if index % 5 == 0:
            bucket_1.append(item)
        elif index % 5 == 1:
            bucket_2.append(item)
        elif index % 5 == 2:
            bucket_3.append(item)
        elif index % 5 == 3:
            bucket_4.append(item)
        else:
            bucket_5.append(item)
    
    df_subset_1 = df.iloc[bucket_1,:]
    df_subset_2 = df.iloc[bucket_2,:]
    df_subset_3 = df.iloc[bucket_3,:]
    df_subset_4 = df.iloc[bucket_4,:]
    df_subset_5 = df.iloc[bucket_5,:]

    return [df_subset_1, df_subset_2, df_subset_3, df_subset_4, df_subset_5]


glass_folds = cross_validation(glass_df)
normalized_glass_folds = cross_validation(normalized_glass_df)

concrete_folds = cross_validation(concrete_df)
normalized_concrete_folds = cross_validation(normalized_concrete_df)


from math import sqrt
from statistics import mode
from statistics import StatisticsError

def knn_classification(train_df, test_df, weighted=False, k=3):
    dropped_train_df = train_df.drop(list(test_df.index.values)).to_numpy()
    test_df = test_df.to_numpy()
    correct_guesses = 0
    wrong_guesses = 0
    
    for i in range(test_df.shape[0]):
        test_row = test_df[i]
        list_of_distances = list()
        for j in range(dropped_train_df.shape[0]):
            train_row = dropped_train_df[j]
            difference = abs(test_row[:-1] - train_row[:-1])
            distance = np.sqrt(sum(np.square(difference)))
            list_of_distances.append([distance, train_row[-1]])
        list_of_distances.sort(key=lambda x: x[0]) # sort by euclidean distances ascending



        if weighted:
            freqs = [0 for i in range(20)]
            guess = -1
            for i in list_of_distances[:k]:
                try:
                    freqs[int(i[1])] += (1 / i[0])
                except ZeroDivisionError: # no distance between the test point and the training point
                    guess = freqs[int(i[1])]
                    break
            guess = np.argmax(freqs) if guess == -1 else guess


        else: # if uniform weighted
            neighbors = list()
            for i in range(k):
                neighbors.append(list_of_distances[i][1])
                try:
                    guess = mode(neighbors)
                except StatisticsError: # all values are distinct in the list
                    guess = neighbors[0]

        if test_row[-1] == guess:
            correct_guesses += 1
        else:
            wrong_guesses += 1

    accuracy = 100*correct_guesses/(correct_guesses + wrong_guesses)
    print("Accuracy for KNN with k={} is {:.2f}%".format(k, accuracy))
    return accuracy




def knn_regression(train_df, test_df, weighted=False, k=3):
    dropped_train_df = train_df.drop(list(test_df.index.values)).to_numpy()
    test_df = test_df.to_numpy()
    
    error = 0
    attempts = 0
    for i in range(test_df.shape[0]):
        test_row = test_df[i]
        list_of_distances = list()
        for j in range(dropped_train_df.shape[0]):
            train_row = dropped_train_df[j]
            difference = abs(test_row[:-1] - train_row[:-1])
            distance = np.sqrt(sum(np.square(difference)))
            list_of_distances.append([distance, train_row])

        list_of_distances.sort(key=lambda x: x[0]) # sort by euclidean distances ascending  


        if weighted:
            
            guess = 0
            total = 0
            weights = 0
            for item in list_of_distances[:k]:
                try:
                    if item[0] == 0:
                        val = 1
                    else:
                        val = item[0]
                    weight = 1/val
                    pull = item[1][-1] * weight
                    total += pull
                    weights += weight

                except ZeroDivisionError: # no distance between the test point and the training point
                    guess = test_row[-1]
                    weight = 0
                    break
            guess = total / weights if guess == 0 else guess
            # print("guess: ", guess)

            
        else: # not weighted
            guess = 0
            for item in list_of_distances[:k]:           
                guess += item[1][-1]
            
            guess /= k
        error += abs(guess - test_row[-1])
        attempts += 1

    mae = error/attempts
    
    print("Mean Absolute Error for KNN with k={}: {}".format(k, mae))
    return mae



k_vals = [1, 3, 5, 7, 9]


for k in k_vals:
    knn_classification_accuracies = list()
    print("KNN Accuracies for Glass Classification without min-max normalization, k={}".format(k))
    for index, i in enumerate(glass_folds):
        print("Fold #{}:".format(index+1))
        knn_classification_accuracies.append(knn_classification(glass_df, i, k=k))
    print("Average KNN Accuracy for Glass Classification without min-max normalization, k={}: {:.2f}%\n-----\n\n".format(k, sum(knn_classification_accuracies) / len(glass_folds)))


for k in k_vals:
    knn_classification_accuracies = list()
    print("KNN Accuracies for Glass Classification with min-max normalization, k={}".format(k))
    for index, i in enumerate(normalized_glass_folds):
        print("Fold #{}:".format(index+1))
        knn_classification_accuracies.append(knn_classification(normalized_glass_df, i, k=k))
    print("Average KNN Accuracy for Glass Classification with min-max normalization, k={}: {:.2f}%\n-----\n\n".format(k, sum(knn_classification_accuracies) / len(normalized_glass_folds)))


for k in k_vals:
    knn_classification_accuracies = list()
    print("Weighted KNN Accuracies for Glass Classification without min-max normalization, k={}".format(k))
    for index, i in enumerate(glass_folds):
        print("Fold #{}:".format(index+1))
        knn_classification_accuracies.append(knn_classification(glass_df, i, weighted=True, k=k))
    print("Average Weighted KNN Accuracy for Glass Classification without min-max normalization, k={}: {:.2f}%\n-----\n\n".format(k, sum(knn_classification_accuracies) / len(glass_folds)))


for k in k_vals:
    knn_classification_accuracies = list()
    print("Weighted KNN Accuracies for Glass Classification with min-max normalization, k={}".format(k))
    for index, i in enumerate(normalized_glass_folds):
        print("Fold #{}:".format(index+1))
        knn_classification_accuracies.append(knn_classification(normalized_glass_df, i, k=k, weighted=True))
    print("Average Weighted KNN Accuracy for Glass Classification with min-max normalization, k={}: {:.2f}%\n-----\n\n".format(k, sum(knn_classification_accuracies) / len(normalized_glass_folds)))


for k in k_vals:
    mae_list = list()
    print("KNN Mean Absolute Error for Concrete Strength Estimation without min-max normalization, k={}".format(k))
    for index, i in enumerate(concrete_folds):
        print("Fold #{}:".format(index+1))
        mae_list.append(knn_regression(concrete_df, i, k=k))
    print("Average KNN Mean Absolute Error for Concrete Strength Estimation without min-max normalization, k={}: {:.2f}".format(k, sum(mae_list) / len(concrete_folds)))


for k in k_vals:
    mae_list = list()
    print("KNN Mean Absolute Error for Concrete Strength Estimation with min-max normalization, k={}".format(k))
    for index, i in enumerate(normalized_concrete_folds):
        print("Fold #{}:".format(index+1))
        mae_list.append(knn_regression(normalized_concrete_df, i, k=k))
    print("Average KNN Mean Absolute Error for Concrete Strength Estimation with min-max normalization, k={}: {:.2f}".format(k, sum(mae_list) / len(normalized_concrete_folds)))


for k in k_vals:
    mae_list = list()
    print("Weighted KNN Mean Absolute Error for Concrete Strength Estimation without min-max normalization, k={}".format(k))
    for index, i in enumerate(concrete_folds):
        print("Fold #{}:".format(index+1))
        mae_list.append(knn_regression(concrete_df, i, k=k, weighted=True))
    print("Average Weighted KNN Mean Absolute Error for Concrete Strength Estimation without min-max normalization, k={}: {:.2f}".format(k, sum(mae_list) / len(concrete_folds)))


for k in k_vals:
    mae_list = list()
    print("Weighted KNN Mean Absolute Error for Concrete Strength Estimation with min-max normalization, k={}".format(k))
    for index, i in enumerate(normalized_concrete_folds):
        print("Fold #{}:".format(index+1))
        mae_list.append(knn_regression(normalized_concrete_df, i, k=k, weighted=True))
    print("Average Weighted KNN Mean Absolute Error for Concrete Strength Estimation with min-max normalization, k={}: {:.2f}".format(k, sum(mae_list) / len(normalized_concrete_folds)))