"""
This module gives an example of how to calculate the mean centerred ratings.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from surprise import Dataset, Reader
from numpy import *
from six import iteritems

# Calculate mean centerred ratings
def mean_centerred_ratings(dataset_path):
    file_path = dataset_path
    reader = Reader(line_format= 'user item rating', rating_scale=(1, 7), sep='\t')
    data = Dataset.load_from_file(file_path, reader)

# Construct a training set using the entire dataset (without spliting the dataset into folds)
# variable trainset is an object of Trainset
    trainset = data.build_full_trainset()

# Parameters needed to calculate mean centerred ratings
    number_users = trainset.n_users
    number_items = trainset.n_items
    user_ratings = trainset.ur
    item_ratings = trainset.ir

    n_x = number_users
    n_y = number_items

    u_matrix = zeros((n_x, n_y), float)
    avg_rating = zeros((n_x, 3), float)
    avg_matrix = zeros((n_x, n_y), float)

# Create original user rating matrix
    for y, y_ratings in iteritems(user_ratings):
        for xi, ri in y_ratings:
            if ri == 0:
                u_matrix[y, xi] += 0
            else:
        	    u_matrix[y, xi] += ri
        	    avg_rating[y, 1] += ri
        	    avg_rating[y, 2] += 1
    print ('Original user ratings\n', u_matrix, '\n')

# Create mean centerred rating matrix
    for y, y_ratings in iteritems(user_ratings):
        for xi, ri in y_ratings:
            if ri == 0:
                avg_matrix[y, xi] += 0
            else:
        	    avg_matrix[y, xi] += avg_rating[y, 1]/avg_rating[y, 2]
	
    mean_matrix = u_matrix-avg_matrix
	
    return mean_matrix
    
# Assign the custom dataset table 2.1
file_path = 'data/table2.1_ratings.txt'

print ('The mean centerred ratings\n', mean_centerred_ratings(file_path))
