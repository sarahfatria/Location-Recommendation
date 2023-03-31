import numpy as np
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import multiprocessing as mp
from multiprocessing import Pool
import time
import math
from math import radians, sin, cos, sqrt, atan2
from scipy.sparse import csr_matrix
import pandas as pd
from collections import defaultdict

def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_relations = []
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_relations.append([uid1, uid2])
    return social_relations


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos

# hasilnya adalah TRAIN set yang terdiri dari uid, lid, dan ctime
def read_training_check_ins(training_matrix):
    check_in_data = open(check_in_file, 'r').readlines()
    training_check_ins = defaultdict(list)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not training_matrix[uid, lid] == 0:
            training_check_ins[uid].append([lid, ctime])
    return training_check_ins

# haislnya adalah TRAIN set yang terdiri dari uid, lid, dan freq
def read_sparse_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return sparse_training_matrix, training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


data_dir = "/home/m439162/Skripsi/Dataset/Gowalla_1/"
size_file = data_dir + "Gowalla_data_size.txt"
check_in_file = data_dir + "Gowalla_checkins.txt"
train_file = data_dir + "Gowalla_train.txt"
tune_file = data_dir + "Gowalla_tune.txt"
test_file = data_dir + "Gowalla_test.txt"
social_file = data_dir + "Gowalla_social_relations.txt"
poi_file = data_dir + "Gowalla_poi_coos.txt"

user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
user_num, poi_num = int(user_num), int(poi_num)

top_k = 100
#AMC = AdditiveMarkovChain(delta_t=3600*24, alpha=0.05)
sparse_training_matrix, training_tuples = read_sparse_training_data()
training_check_ins = read_training_check_ins(sparse_training_matrix)
sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
                                for uid in training_check_ins}
social_relations = read_friend_data()
ground_truth = read_ground_truth()
poi_coos = read_poi_coos()
all_uids = list(range(user_num))
all_lids = list(range(poi_num))


#AMC.build_location_location_transition_graph(sorted_training_check_ins)

def is_friend(key_id, target_id):
    if target_id in social_dict.get(key_id, []):
        result = 1
    else:
        result = 0
    return result

social_dict = defaultdict(list)
for id1, id2 in social_relations:
    social_dict[id1].append(id2)

user_location_dict = {}
for user_id, location_id in training_tuples:
    if user_id in user_location_dict:
        user_location_dict[user_id].append(location_id)
    else:
        user_location_dict[user_id] = [location_id]

friends_visited_locations = defaultdict(list)
for user in all_uids:
    for friend in social_dict[user]:
        friends_visited_locations[user] = list(set(friends_visited_locations[user]).union(set(user_location_dict[friend])))


def eta(u_prime):
    denominator_sum = 0
    for l_prime in friends_visited_locations[u_prime]:
        inner_sum = 0
        for u_double_prime in social_dict[u_prime]:
            inner_sum += sparse_training_matrix[u_double_prime, l_prime]
            denominator_sum += math.log(inner_sum + 1)
    result_out = open(output_dir + "eta" + ".txt", 'a')
    result_out.write('\t'.join([
        str(u_prime),
        str(denominator_sum)
    ]) + '\n')
    result_out.close()


output_dir = "/home/m439162/Skripsi/Output/Spatio/Gowalla/"
result_out = open(output_dir + "eta" + ".txt", 'w')

with mp.Pool(processes=20) as pool:
        pool.map(eta, [(u_prime) for u_prime in all_uids])

