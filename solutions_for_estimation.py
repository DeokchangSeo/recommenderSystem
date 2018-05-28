"""
This is solutions for week13_prac.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

from collections import defaultdict
from surprise import KNNBasic
from surprise import Dataset, Reader

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

# Create a ground_truth
def ground_truth(testset, threshold = 3):
    modified_testset = []
    for x in testset:
        if x[2] > threshold:
            modified_testset.append((x[0],x[1],1.0))
        else:
            modified_testset.append((x[0],x[1],0.0))
    
    return modified_testset

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Create a set of k-recommended items and a set of positive items
def create_k_reco_posi(prediction):
    # top k recommended items
    k_recommend = {}

    for x, recommendations in prediction.items():
        k_recommend[x] = (list([iid for (iid, _) in recommendations]))

    #print (k_recommend)
    
    # take only positive items    
    p_items = {}
    for uid, recommends in k_recommend.items():
        #print (uid, recommends)
        temp_A = []
        for x in ground_truth(testset, threshold):
            if uid == x[0] and x[2] == 1:
                temp_A.append(x[1])
        p_items[uid] = temp_A
    
    #print (p_items)
    return k_recommend, p_items
    
# Calculate precision
def precision(prediction):
    # prepare the parameters to calculate precision
    k_recommend, p_items = create_k_reco_posi(prediction)
    
    myDict = {}
    print ("\nUser\t k-recommended\t\t True positive items")
    for uid1, recommend in k_recommend.items():
        for uid2, positives in p_items.items():
            if uid1 == uid2:
                print (uid1, '\t', recommend, '\t', positives)
                #print ((len(list(set(recommend) & set(positives))) / len(recommend)) * 100)
                myDict[uid1] = (len(list(set(recommend) & set(positives))) / len(recommend)) * 100
    return myDict
    
# Calculate recall
def recall(prediction):
    # prepare the parameters to calculate precision
    k_recommend, p_items = create_k_reco_posi(prediction)
    
    myDict = {}
    print ("\nUser\t k-recommended\t\t True positive items")
    for uid1, recommend in k_recommend.items():
        for uid2, positives in p_items.items():
            if uid1 == uid2:
                print (uid1, '\t', recommend, '\t', positives)
                #print ((len(list(set(recommend) & set(positives))) / len(positives)) * 100)
                if len(positives) == 0:
                    myDict[uid1] = "Cannot be devided by zero"
                else: myDict[uid1] = (len(list(set(recommend) & set(positives))) / len(positives)) * 100
    return myDict

# Custom dataset table 2.1
#file_path = 'data/table2.1_ratings.txt'
file_path = 'ratings_for_question1.txt'
trainset = create_full_trainset(file_path, line_format= 'user item rating', rating_scale=(1, 7), sep='\t')
testset = trainset.build_testset()

# Question 1 --> create ground truth
threshold = 3
print (ground_truth(testset, threshold))


# Prepare the parameters to calculate precision and recall
algo = KNNBasic()
algo.train(trainset)
predictions = algo.test(testset)
top_n = get_top_n(predictions, n=3)

# Question 2 --> precision
print (precision(top_n))

# Question 3 --> recall
print (recall(top_n))