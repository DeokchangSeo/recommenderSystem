"""

This is an implementation to the basic collaborative filtering method.

"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from surprise import similarities as sims
from surprise import Dataset, Reader
from six import iteritems
from numpy import *
import heapq

# Load and return a custom dataset
# Return: an object of DatasetAutoFolds, dataset.raw_ratings contains raw ratings
def load_dataset(dataset_path,line_format= 'user item rating', rating_scale=(1, 7), sep=None):
    # Define the line format and rating scale in the data file
    name = None
    file_format = Reader(name, line_format, sep, rating_scale)
    
    # Load the dataset and return it
    dataset = Dataset.load_from_file(dataset_path, file_format)
    return dataset


# Create a training set using the entire dataset
def create_full_trainset(dataset_path,line_format= None, rating_scale=(1, 7), sep=None):
    data = load_dataset(dataset_path,line_format, rating_scale, sep)
    # Construct a training set using the entire dataset
    trainset = data.build_full_trainset()
    return trainset


# Return a mean-centered rating matrix created from an input rating matrix
def mean_centered_ratings(rating_matrix):
    myDict = {}
    # calculate mean centered ratings
    for x, x_ratings in iteritems(rating_matrix): # for each row in rating_matrix
        temp_a, temp_b = [], []
        temp_a = temp_a + ([yi for xi, yi in x_ratings])
        for xi, yi in x_ratings:
            temp_b.append((xi, round(yi- (sum(temp_a)/len(x_ratings)),2)))
            myDict[x] = temp_b

    return myDict


# Calculate adjusted Cosine Similarity
def adjustedCosine(num_rows, rating_matrix, min_support):
    """
        Compute the adjusted cosine similarity between all pairs of rows.
        Only **common** columns are taken into account.
    """
    myDict = mean_centered_ratings(rating_matrix)    

    # parameters to calculate adjusted cosine similarity
    prods = zeros((num_rows, num_rows), double)
    freq = zeros((num_rows, num_rows), double)
    sqi = zeros((num_rows, num_rows), double)
    sqj = zeros((num_rows, num_rows), double)
    sim = zeros((num_rows, num_rows), double)

    # calculation of adjusted cosine similarity
    for y, y_ratings in iteritems(myDict):
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(num_rows):
        sim[xi, xi] = 1
        for xj in range(xi + 1, num_rows):
            if freq[xi, xj] < min_support:
                sim[xi, xj] = 0
            else:
                denum = sqrt(sqi[xi, xj]) * sqrt(sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]
    return sim


# Calculate collaborative filtering prediction
def rating_prediction(sim, target, rTarget, tRatings, neighbor_size):
    # Select similarity for the users/items in the tRatings
    target_sim = []
    for iID, rtings in tRatings[rTarget]:
        target_sim.append(sim[target][iID])
    
    # Select the highest neighbor_size similarities for target
    n_sims = heapq.nlargest(neighbor_size, target_sim)

    # Get the neighbors id, sim(target, id) is in n_sims 
    neighbours = {}
    for x in range(neighbor_size):
        i = 0
        for y in sim[target]:
            if y == n_sims[x] and i != target:
                neighbours[i] = y
            i += 1

    # Select ratings from raw_ratings
    selected_ratings = {}
    for x in list(neighbours.keys()):
        for iID, rating in tRatings[rTarget]:
            if x == iID:
                selected_ratings[x] = rating
    
    #print('selected_ratings ', selected_ratings)
    
    # if the number of selected ratings is less than neighbor_size, return 0. No sufficient neighbors
    if len(selected_ratings) < neighbor_size:
        print('No sufficient neighbors')
        return 0

    # Calculate the prediction based on neighbours' ratings
    prediction_ratings = 0
    for x in list(neighbours.keys()):
        #print (selected_ratings[x], '\t', neighbours[x])
        prediction_ratings += selected_ratings[x] * neighbours[x]

    return prediction_ratings / sum(list(neighbours.values()))


def prediction_by_CF(trainset, item_rawId, user_rawId, neighhor_size, user_item, sim_method):
# trainset: rating dataset which is a Trainset object
# item_rawId, user_rawId: raw ids for an item and a user
# neighhor_size: number of nearest neighbors used to calculate the predicted rating
# user_item: 'user' or 'item', use user_based CF if it is 'user', otherwise use item_based CF
# sim_method: three values: 'cosine', 'pearson', 'adjustedCosine' 
# Return: a predicted rating to item_rawId for user_rawId by using the basic collaborative filtering method

    # Inner ids
    itemId = item_rawId - 1
    userId = user_rawId - 1
    min_support = 0

    number_users = trainset.n_users
    number_items = trainset.n_items
    user_ratings = trainset.ur
    item_ratings = trainset.ir

    # Configure parameters  for  user-based
    n_target, target, rTarget, ratings, tRatings = number_users, userId, itemId, user_ratings, item_ratings

    # Configure parameters for item-based
    if user_item != "user":
        n_target, target, rTarget, ratings, tRatings = number_items, itemId, userId, item_ratings, user_ratings
    
    args = [n_target, tRatings, min_support]

    # choose similarity method
    if sim_method == "cosine":
        sim = sims.cosine(*args)
    elif sim_method == "pearson":
        sim = sims.pearson(*args)
    else: 
        sim = adjustedCosine(*args)

    print('\nSimilarity:\n', sim)
    
    return rating_prediction(sim, target, rTarget, tRatings, neighhor_size)
 

# Custom dataset table 2.1
#file_path = 'data/table2.1_ratings.txt'
file_path = 'ratings_for_question1.txt'
trainset = create_full_trainset(file_path, line_format= 'user item rating', rating_scale=(1, 7), sep='\t')

# trainset.ur contains user ratings
print("Original user ratings\nUser \t Ratings (item_id, rating)")
for user_id, item_ratings in trainset.ur.items():
	 print(user_id, '\t', item_ratings)


# Parameter setting, itemID, userID, neighbour_size, user-item, similarity_method
item_rawId = 1
user_rawId = 3
neighbor = 2
user_item = 'user'
sim_method = 'pearson'
estimate = prediction_by_CF(trainset, item_rawId, user_rawId, neighbor, user_item, sim_method)
print ('\nUsing {} based CF, the predicted rating to item {} for user {} is: {est:4.2f} \n'.format(user_item, item_rawId, user_rawId, est=estimate))

 
item_rawId = 6
estimate = prediction_by_CF(trainset, item_rawId, user_rawId, neighbor, user_item, sim_method)
print ('\nUsing {} based CF, the predicted rating to item {} for user {} is: {est:4.2f} \n'.format(user_item, item_rawId, user_rawId, est=estimate))

item_rawId = 1
user_rawId = 3
neighbor = 2
user_item = 'item'
sim_method = 'adjustedCosine'
estimate = prediction_by_CF(trainset, item_rawId, user_rawId, neighbor, user_item, sim_method)
print ('\nUsing {} based CF, the predicted rating to item {} for user {} is: {est:4.2f} \n'.format(user_item, item_rawId, user_rawId, est=estimate))

item_rawId = 6
estimate = prediction_by_CF(trainset, item_rawId, user_rawId, neighbor, user_item, sim_method)
print ('\nUsing {} based CF, the predicted rating to item {} for user {} is: {est:4.2f} \n'.format(user_item, item_rawId, user_rawId, est=estimate))
 
item_rawId = 3
user_rawId = 2
estimate = prediction_by_CF(trainset, item_rawId, user_rawId, neighbor, user_item, sim_method)
print ('\nUsing {} based CF, the predicted rating to item {} for user {} is: {est:4.2f} \n'.format(user_item, item_rawId, user_rawId, est=estimate))
