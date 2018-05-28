"""
This module gives an example of how to calculate the mean centerred ratings.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from surprise import Dataset, Reader
from numpy import *
from six import iteritems

# Compare two vectors. (only for common elements)
# If any element contains 0, then delete that based on index.
def compare_elements(element1, element2):
    temp = []
    
    for i in range(len(element1)):
        if (element1[i] == 0) or (element2[i] == 0):
            temp.append(i)

    element1 = delete(element1, temp)
    element2 = delete(element2, temp)
    print ('Sorted to calculate similarity:\n', element1, '\n', element2, '\n')
    return element1, element2

# Compare two vectors. (use all elements)
# If any element contains 0, then delete that based on index.
def compare_elements_all(element1, element2):
    temp1, temp2 = [], []
    
    for i in range(len(element1)):
        if (element1[i] == 0):
            temp1.append(i)
        if (element2[i] == 0):
            temp2.append(i)

    element1 = delete(element1, temp1)
    element2 = delete(element2, temp2)
    print ('Sorted to calculate average:\n', element1, '\n', element2, '\n')
    return element1, element2

# Cosine similarity 
def sim_cosine(element1, element2):
    """
        consine_sim(u,v) = sumxy / (sqrt(sumxx) * sqrt(sumyy))
        k = 1 ~ len(element1)
        sumxy = (x1*y1) + ... + (xk*yk)
        sumxx = x^2 + ... + x^k
        sumyy = y^2 + ... + y^k
    """

# Parameters needed to calculate cosine similarity
    a, b = compare_elements(element1, element2)
    assert len(a) == len(b)
    n = len(a)
    assert n > 0
    sumxx, sumxy, sumyy = 0, 0, 0

# Calculate cosine similarity between 2 items or users.
    for i in range(len(a)):
        x = a[i]
        y = b[i]

        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy / (math.sqrt(sumxx) * math.sqrt(sumyy))

# Pearson similarity
def sim_pearson(element1, element2):
    """
        pearson_sim(u,v) = sumxy_avg / (sqrt(sumxx_avg) * sqrt(sumyy_avg))
        k = 1 ~ len(element1)
        sumxy_avg = sum((x-avg_x)*(y-avg_y))
        sumxx_avg = sum(x-avg_x)^2
        sumyy_avg = sum(y-avg_y)^2
    """
   
# Parameters needed to calculate pearson similarity 
    c, d = compare_elements_all(element1, element2)
    a, b = compare_elements(element1, element2)
    assert len(a) == len(b)
    n = len(a)
    assert n > 0
    avg_x = sum(c) / len(c)
    avg_y = sum(d) / len(d)
    sumxy_avg, sumxx_avg, sumyy_avg = 0, 0, 0

# Calculate pearson similarity between 2 items or users.
    for i in range(n):
        adiff = a[i] - avg_x
        bdiff = b[i] - avg_y
        sumxy_avg += adiff * bdiff
        sumxx_avg += adiff * adiff
        sumyy_avg += bdiff * bdiff

    return sumxy_avg / (math.sqrt(sumxx_avg) * math.sqrt(sumyy_avg))

# Calculate any pair of users or items with given similarity measures.
def similarity_for_pair(dataset_path, element1, element2, users, cosine):
    file_path = dataset_path
    reader = Reader(line_format= 'user item rating', rating_scale=(1, 7), sep='\t')
    data = Dataset.load_from_file(file_path, reader)

# Construct a training set using the entire dataset (without spliting the dataset into folds)
# variable trainset is an object of Trainset
    trainset = data.build_full_trainset()

# Parameters needed to create ratings matrix
    number_users = trainset.n_users
    number_items = trainset.n_items
    user_ratings = trainset.ur
    item_ratings = trainset.ir
    user_item = "User"
    cosine_pearson = "Cosine"

# Create empty tables based on user input
    if users == True:
        ratings = user_ratings
        matrix = zeros((number_users, number_items), float)
    else:
        user_item = "Item"
        ratings = item_ratings
        matrix = zeros((number_items, number_users), float)

# Create user or item rating matrix
    for y, y_ratings in iteritems(ratings):
        for xi, ri in y_ratings:
            if ri == 0:
                matrix[y, xi] += 0
            else:
        	    matrix[y, xi] += ri

    print ('Original table:\n', matrix, '\n')
    print ('Picked:\n', matrix[element1], '\n', matrix[element2], '\n')

# Apply cosine or pearson similarity to a pair of users or items
    if cosine == True:
        sim = sim_cosine(matrix[element1], matrix[element2])
    else:
        cosine_pearson = "Pearson"
        sim = sim_pearson(matrix[element1], matrix[element2])

    return ('{} similarity between {} {} and {} is: {}'.format(cosine_pearson, user_item, element1, element2, sim))


# Assign the custom dataset table 2.1
file_path = 'data/table2.1_ratings.txt'
users = True
cosine = True

print (similarity_for_pair(file_path, 0, 2, users, cosine))