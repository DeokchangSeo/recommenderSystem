"""
This is solutions for question sheet.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from surprise import KNNBaseline
from surprise import similarities as sims
from surprise import Dataset, Reader
from numpy import *
from six import iteritems
import warnings
import os

# Create dataset and make matrix
def create_dataset(dataset_path):
    # Read the data file
    file_path = dataset_path
    reader = Reader(line_format= 'user item rating', rating_scale=(1, 7), sep='\t')
    data = Dataset.load_from_file(file_path, reader)

    # Construct a training set using the entire dataset (without spliting the dataset into folds)
    # variable trainset is an object of Trainset
    trainset = data.build_full_trainset()

    # Parameters needed to create rating matrix
    number_users = trainset.n_users
    number_items = trainset.n_items
    user_ratings = trainset.ur
    item_ratings = trainset.ir

    return number_users, number_items, user_ratings, item_ratings
    
# Create original user rating matrix
def create_orginal_rating_matrix(dataset_path):
    # Parameters needed to create rating matrix
    number_users, number_items, user_ratings, item_ratings = create_dataset(dataset_path)

    u_matrix = zeros((number_users, number_items), float)
    avg_rating = zeros((number_users, 3), float)
    avg_matrix = zeros((number_users, number_items), float)

    # Create rating matrix
    for y, y_ratings in iteritems(user_ratings):
        for xi, ri in y_ratings:
            if ri == 0:
                u_matrix[y, xi] += 0
            else:
        	    u_matrix[y, xi] += ri
        	    avg_rating[y, 1] += ri
        	    avg_rating[y, 2] += 1
        	    
    return u_matrix, avg_rating, avg_matrix

# Calculate and create mean centered rating matrix
def mean_centered_matrix(dataset_path):
    # Parameters needed to create rating matrix 
    number_users, number_items, user_ratings, item_ratings = create_dataset(dataset_path)
    u_matrix, avg_rating, avg_matrix = create_orginal_rating_matrix(dataset_path)

    # Create mean centered rating matrix
    for y, y_ratings in iteritems(user_ratings):
        for xi, ri in y_ratings:
            if ri == 0:
                avg_matrix[y, xi] += 0
            else:
        	    avg_matrix[y, xi] += avg_rating[y, 1]/avg_rating[y, 2]
	
    mean_matrix = u_matrix-avg_matrix
    return mean_matrix

# Similarity calculation
def cosine_similarity_users(dataset_path, users, cosine):
    # Parameters needed for similarity
    number_users, number_items, user_ratings, item_ratings = create_dataset(dataset_path)
    min_support = 0
    
    # Configure user-user cosine similarity
    args = [number_users, item_ratings, min_support]
    user_csim = sims.cosine(*args)
    
    # Configure user-user pearson similarity
    args = [number_users, item_ratings, min_support]
    user_psim = sims.pearson(*args)
    
    # Calculate item-item cosine correlation similarity
    args = [number_items, user_ratings, min_support]
    item_csim = sims.cosine(*args)

    # Calculate item-item pearson correlation similarity
    args = [number_items, user_ratings, min_support]
    item_psim = sims.pearson(*args)

    if users == True:
        if cosine == True:
            return user_csim
        else: return user_psim
    else:
        if cosine == True:
            return item_csim
        else: return item_psim

# Calculate similarity for a pair of users or items
def similarity_pair_elements(element1, element2, dataset_path, users, cosine):
    # Parameters needed to calculate similarity
    sim = cosine_similarity_users(dataset_path, users, cosine)
    pair = sim[element1-1,element2-1]
    user = "user"
    cos = "cosine"
    
    if users != True:
        user = "item"
    
    if cosine != True:
        cos = "pearson"
    
    return '\nThe {} similarity between {} {} and {} {}: {}'.format(cos, user, element1, user, element2, pair)

# Create mean centered rating list to calculate adjusted cosine similarity
def create_mcr_list(num_users, num_items, mcr):
    # Parameters needed to create a list
    num_row = len(mcr)
    num_col = len(mcr[0])
    list = {}
    temp_table = []
        
    # Create a list of ratings to calculate similarity
    for x in mcr:
        temp_row = []
        for y in range(num_col):
            if x[y] != 0:
                temp_element = (y, x[y])
                temp_row.append(temp_element)
            
        temp_table.append(temp_row)
    
    for x in range(num_row):
        list[x] = temp_table[x]
 
    return num_col, list

# Calculate and create adjusted cosine similarity
def adjusted_cosine_similarity(dataset_path):
    # Calculate and create mean centered rating matrix
    mcr = mean_centered_matrix(file_path)
    number_users, number_items, user_ratings, item_ratings = create_dataset(dataset_path)
    num_col, list = create_mcr_list(number_users, number_items, mcr)
    min_support = 0
    
    args = [num_col, list, min_support]
    sim = sims.cosine(*args)
    return sim

# Calculate collaborative filtering prediction
def prediction_on_CF(dataset_path, options):
    # Parameters needed to calculate prediction
    itemId = options[0] - 1 
    userId = options[1] - 1
    base = options[2]
    sim_method = options[3]
    min_support = 0

    number_users, number_items, user_ratings, item_ratings = create_dataset(dataset_path)
    target = userId
    rTarget = itemId
    #neighbours = range(0, number_users)
    ratings = user_ratings
    args = [number_users, item_ratings, min_support]

    # Configure parameters
    if base != "user":
        args = [number_items, user_ratings, min_support]
        #neighbours = range(0, number_items)
        ratings = item_ratings
        target = itemId
        rTarget = userId

    if sim_method == "pearson":
        sim = sims.pearson(*args)
    else:
        mcr = mean_centered_matrix(dataset_path)
        num_col, list = create_mcr_list(number_users, number_items, mcr)
        obsolete, ratings = create_mcr_list(number_users, number_items, transpose(mcr))
        args = [num_col, list, min_support]

        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sim = sims.cosine(*args)

    neighbourA = target - 1
    neighbourB = target + 1
    
    sim_A = sim[neighbourA, target]
    sim_B = sim[target, neighbourB]
    
    # Calculate prediction for an item
    selected_ratings = []
    for uID, y_rating in iteritems(ratings):
        if uID == neighbourA or uID == neighbourB:
            print (uID, y_rating)
            for iID, rating in y_rating:
                if iID == rTarget:
                    selected_ratings.append(rating)

    if len(selected_ratings) != 2:
        return 0

    return ((selected_ratings[0]*sim_A) + (selected_ratings[1]*sim_B)) / (sim_A + sim_B)

# k-nearest neighbours of a user using the KNNBasic with cosine similarity
def knn_cosine(dataset_path, target, value):
    # Read the data file
    file_path = dataset_path
    reader = Reader(line_format= 'user item rating', rating_scale=(1, 7), sep='\t')
    data = Dataset.load_from_file(file_path, reader)

    # Construct a training set using the entire dataset (without spliting the dataset into folds)
    # variable trainset is an object of Trainset
    trainset = data.build_full_trainset()
    
    # Parameters needed to create rating matrix
    user_to_item = {}
    item_to_user = {}
    file = open(file_path, "r")
        
    # Train the algorithm to compute the similarities between users
    sim_options = {'name': 'cosine', 'user_based': True}
    algo = KNNBaseline(sim_options=sim_options)
    algo.train(trainset)
    
    # Read the mappings user <-> item
    for line in file:
        line = line.split('\t')
        user_to_item[line[0]] = line[1]
        item_to_user[line[1]] = line[0]
    
    # Retrieve the user id and 
    target_neighbors = algo.get_neighbors(target, k=value)
    target_neighbors = (algo.trainset.to_raw_uid(inner_id) for inner_id in target_neighbors)
    target_neighbors = (item_to_user[rid] for rid in target_neighbors)
                       
    return target_neighbors

# Cosine similarity 
def sim_cosine(vector1, vector2):
    """
        consine_sim(u,v) = sumxy / (sqrt(sumxx) * sqrt(sumyy))
        k = 1 ~ len(vector1)
        sumxy = (x1*y1) + ... + (xk*yk)
        sumxx = x^2 + ... + x^k
        sumyy = y^2 + ... + y^k
    """

    # Parameters needed to calculate cosine similarity
    assert len(vector1) == len(vector2)
    n = len(vector1)
    assert n > 0
    sumxx, sumxy, sumyy = 0, 0, 0

    # Calculate cosine similarity between 2 vectors
    for i in range(len(vector1)):
        x = vector1[i]
        y = vector2[i]

        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    if sumxx == 0 or sumyy == 0 :
        return '0\nsim_cosine: Invalid vector values. 0 denominator occurs.'
    
    return sumxy / (math.sqrt(sumxx) * math.sqrt(sumyy))

# Pearson similarity
def sim_pearson(vector1, vector2):
    """
        pearson_sim(u,v) = sumxy_avg / (sqrt(sumxx_avg) * sqrt(sumyy_avg))
        k = 1 ~ len(element1)
        sumxy_avg = sum((x-avg_x)*(y-avg_y))
        sumxx_avg = sum(x-avg_x)^2
        sumyy_avg = sum(y-avg_y)^2
    """
   
    # Parameters needed to calculate pearson similarity 
    assert len(vector1) == len(vector2)
    n = len(vector1)
    assert n > 0
    avg_x = sum(vector1) / len(vector1)
    avg_y = sum(vector2) / len(vector2)
    sumxy_avg, sumxx_avg, sumyy_avg = 0, 0, 0

    # Calculate pearson similarity between 2 vectors.
    for i in range(n):
        adiff = vector1[i] - avg_x
        bdiff = vector2[i] - avg_y
        sumxy_avg += adiff * bdiff
        sumxx_avg += adiff * adiff
        sumyy_avg += bdiff * bdiff

    if sumxx_avg == 0 or sumxx_avg == 0 :
        return '0\nsim_pearson: Invalid vector values. 0 denominator occurs.'
    
    return sumxy_avg / (math.sqrt(sumxx_avg) * math.sqrt(sumyy_avg))

# Assign the custom dataset table 2.1
file_path = 'data/table2.1_ratings.txt'

# Show original rating matrix
print ('Original user ratings\n', create_orginal_rating_matrix(file_path)[0])

# Question 1
users = True
cosine = True
print ('\nQuestion 1\nSimilarity matrix\n', cosine_similarity_users(file_path, users, cosine))
element1 = 1
element2 = 3
print (similarity_pair_elements(element1, element2, file_path, users, cosine))

# Question 2 - mean centered ratings
print ('\nQuestion 2\nMean centered ratings\n', mean_centered_matrix(file_path))

# Question 3 - adjusted cosine similarity
print ('\nQuestion 3\nAdjusted cosine similarity\n', adjusted_cosine_similarity(file_path))

# Question 4 - Predict the absolute rating of item 3 for user 3
# Configure option: [itemID, userID, user-based/item-based, pearson/adjustedCosine]
options = [3, 2, 'user', 'pearson']
# (a) user-based collaborative filtering with Pearson correlation
print ('\nQuestion 4\n(a) User-based collaborative filtering with Pearson correlation')
print ('Prediction based on raw rating of item {} for user {} is: {} '.format(options[0], options[1], prediction_on_CF(file_path, options)))
# (b) item-based collaborative filtering with adjusted cosine similarity
options[2] = 'item'
options[3] = 'adjustedCosine'
print ('\n(b) Item-based collaborative filtering with adjusted cosine similarity')
print ('Prediction based on raw rating of item {} for user {} is: {} '.format(options[0], options[1], prediction_on_CF(file_path, options)))

# Question 5
new_file_path = 'data/ratings_for_q5.txt'
print ('\nQuestion 5\nNew User ratings\n', create_orginal_rating_matrix(new_file_path)[0])
options = [3, 2, 'user', 'pearson']
# (a) user-based collaborative filtering with Pearson correlation with new file
print ('\n(a) User-based collaborative filtering with Pearson correlation')
print ('User-based collaborative filtering with Pearson correlation: {} '.format(prediction_on_CF(new_file_path ,options)))
# (b) item-based collaborative filtering with adjusted cosine similarity
options[2] = 'item'
options[3] = 'adjustedCosine'
print ('\n(b) Item-based collaborative filtering with adjusted cosine similarity')
print ('Prediction based on raw rating of item {} for user {} is: {} \n'.format(options[0], options[1], prediction_on_CF(new_file_path, options)))

# Question 6
k = 3
user = 2
print ('\nQuestion 6\nThe 3 nearest neighbors of user 2 are: {}'.format(list(knn_cosine(file_path, user, k))))

# Question 7
vector1 = [1,1,1]
vector2 = [0,0,0]
print ('\nQuestion 7\nCosin similarity between vector {} and vector {} is: {}'.format(vector1, vector2, sim_cosine(vector1, vector2)))
vector1 = [6,5,4,6,3,2]
vector2 = [2,3,3,1,5,4]
print ('\nPearson similarity between vector {} and vector {} is: {}'.format(vector1, vector2, sim_pearson(vector1, vector2)))
