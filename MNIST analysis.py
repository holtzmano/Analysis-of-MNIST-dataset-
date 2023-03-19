# Oren Holtzman I.D: 209905207
# Ishay Post I.D: 205415607

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import load_digits
from scipy.spatial import distance
from sklearn.datasets import fetch_openml
from sklearn import metrics, preprocessing
# from statistics import mode
from sklearn import linear_model
import math


def make_avg_digit(x, y):
    """
    Builds an average digit matrix, of the matricies that correspond to each digit
    Parameters
    ----------
    x - Image matricies, y - Predicted labels
    Returns
    -------
    same_predicted_x - a matrix that for each row it has all images in that row
                       are classified as that row number
             avg_x - a numpy array of average matricies
    """

    identical_predicted_y_labels = [np.where(y == i) for i in range(10)]
    same_predicted_x = [x[identical_predicted_y_labels[i]] for i in range(10)]
    avg_x = [np.average(same_predicted_x[i], axis=0) for i in range(10)]

    return same_predicted_x, avg_x


def show_average_digits(avg_mat):
    """
    Plots the average digits of avg_mat for each digit
    Parameters
    ----------
    avg_mat - numpy array of average matricies
    Returns
    -------
    None.
    """
    reshape_value = int(math.sqrt(avg_mat[0].shape[0]))

    for i in range(10):
        plt.figure()
        plt.subplots_adjust()
        plt.imshow(avg_mat[i].reshape(reshape_value, reshape_value), cmap=plt.cm.gray)
        plt.title(i, size=20)
        plt.xticks(())
        plt.yticks(())
        plt.show()


def dist(a, b):
    """
    Calculates cosine distance
    Parameters
    ----------
    Two vectors - a, b
    Returns
    -------
    Cosine distance of a and b
    """

    return distance.cosine(np.array(a).flatten(), np.array(b).flatten())


def calculate_distance_matrix(same_predicted_x, average_x):
    """
    Calculates the distance for each digit's images from the average image of that digit
    Parameters
    ----------
        same_predicted_x - Matrix that for each row it holds it has the images
                        that are classified to that corresponding row number
         average_x - average digit values matricies
    Returns
    -------
    matrix_distances - Has the distances of every digit's image of that class(digit #)
                       from the average image of that class
    """
    return [list(map(lambda x: dist(average_x[i], x), same_predicted_x[i])) for i in range(10)]


def show_distance_histograms(matrix_distances):
    """
    Plots histograms of matrix_distances, plots lines of the median, mode
    and mean values of the data in mat_distances

    Parameters
    ----------
    matrix_distances - Has the distances of every digit's image of that class(digit #)
                       from the average image of that class
    Returns
    -------
    None.
    """

    for i in range(10):
        plt.hist(matrix_distances[i], density=True, bins=25, histtype='barstacked', edgecolor='black')
        plt.title(f"Digit # {i}")
        plt.axvline(np.median(matrix_distances[i]), color='gold', label='Median')
        plt.axvline(max(set(matrix_distances[i]), key=matrix_distances[i].count), color='silver', label='Mode')
        plt.axvline(np.mean(matrix_distances[i]), color='red', label='Mean')
        plt.legend()
        plt.show()


def classify_image_by_distance_of_avg(average_x, image):
    return np.argmin([dist(i, image) for i in average_x])


def classify_data_by_distance_of_avg(x, average_x):
    return [classify_image_by_distance_of_avg(average_x, image) for image in x]


def show_confusion_matrix(actual, predicted):
    '''
    Prints confusion matrix
    '''
    classification_report = metrics.classification_report(actual, predicted)
    print(f"Classification Report:\n{classification_report}\n")
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    print(f"Confusion Matrix:\n{confusion_matrix}")


# Solution


digits = load_digits()
x = digits.data
y = digits.target

# Question 2A:
same_predicted_x, average_x = make_avg_digit(x, y)
show_average_digits(average_x)

# Question 2B:
matrix_distances = calculate_distance_matrix(same_predicted_x, average_x)

# Question 2C:
show_distance_histograms(matrix_distances)

# Question 2D:
y_prediction = classify_data_by_distance_of_avg(x, average_x)
show_confusion_matrix([int(i) for i in y], y_prediction)

# Question 2E:
distances = [np.array([dist(image, avg) for image in x]) for avg in average_x]
X = np.column_stack((distances[0], distances[1], distances[2],
                     distances[3], distances[4], distances[5],
                     distances[6], distances[7], distances[8], distances[9]))
X_scaled = preprocessing.scale(X)
Y = y
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=200)
logistic_classifier.fit(X_scaled, Y)
expected = Y
predicted = logistic_classifier.predict(X_scaled)
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
print("Logistic Regression Cross Validation Report:\n%s\n" % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))

# Question 2H
x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
# Question 2HA:
same_predicted_x, average_x = make_avg_digit(x, y)
show_average_digits(average_x)
# Question 2HB:
matrix_distances = calculate_distance_matrix(same_predicted_x, average_x)
# Question 2HC:
show_distance_histograms(matrix_distances)
# Question 2HD:
y_prediction = classify_data_by_distance_of_avg(x, average_x)
show_confusion_matrix(y, y_prediction) # [int(i) for i in y]
# Question 2HE:
distances = [np.array([dist(image, avg) for image in x]) for avg in average_x]
X = np.column_stack((distances[0], distances[1], distances[2],
                     distances[3], distances[4], distances[5],
                     distances[6], distances[7], distances[8], distances[9]))
X_scaled = preprocessing.scale(X)
Y = y
logistic_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=200)
logistic_classifier.fit(X_scaled, Y)
expected = Y
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10)
print("Logistic Regression Cross Validation Report:\n%s\n" % (metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))