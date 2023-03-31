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

from lib.AdditiveMarkovChain import AdditiveMarkovChain
from lib.metrics import precisionk, recallk

start_program = time.time()

data_dir = "/home/m439162/Skripsi/Dataset/Yelp_1/"
output_dir = '/home/m439162/Skripsi/Output/Spatio/Yelp/'


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


size_file = data_dir + "Yelp_data_size.txt"
check_in_file = data_dir + "Yelp_check_ins.txt"
train_file = data_dir + "Yelp_train.txt"
tune_file = data_dir + "Yelp_tune.txt"
test_file = data_dir + "Yelp_test.txt"
social_file = data_dir + "Yelp_social_relations.txt"
poi_file = data_dir + "Yelp_poi_coos.txt"

user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
user_num, poi_num = int(user_num), int(poi_num)

top_k = 100
AMC = AdditiveMarkovChain(delta_t=3600*24, alpha=0.05)
sparse_training_matrix, training_tuples = read_sparse_training_data()
training_check_ins = read_training_check_ins(sparse_training_matrix)
sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
                                for uid in training_check_ins}
social_relations = read_friend_data()
ground_truth = read_ground_truth()
poi_coos = read_poi_coos()
all_uids = list(range(user_num))
all_lids = list(range(poi_num))


AMC.build_location_location_transition_graph(sorted_training_check_ins)

time_diffs = []
for uid, check_ins in sorted_training_check_ins.items():
    check_ins.sort(key=lambda x: x[1])  # Sort by time
    diffs = [check_ins[i+1][1] - check_ins[i][1] for i in range(len(check_ins)-1)]
    time_diffs.extend(diffs)

# Calculate alpha
alpha = 1 + len(time_diffs) / sum([math.log(diff+1) for diff in time_diffs])


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # radius of the Earth in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def read_complete_training_check_ins(training_matrix, poi_coos):
    check_in_data = open(check_in_file, 'r').readlines()
    complete_training_check_ins = defaultdict(list)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not training_matrix[uid, lid] == 0:
            lat, lon = poi_coos[lid]
            complete_training_check_ins[uid].append([lid, lat, lon, ctime])
    return complete_training_check_ins


complete_training_check_ins = read_complete_training_check_ins(training_check_ins, poi_coos)


Y = []
for user_check_ins in complete_training_check_ins.values():
    lat_lons = [(lat, lon) for lid, lat, lon, ctime in user_check_ins]
    for i in range(len(lat_lons) - 1):
        lat1, lon1 = lat_lons[i]
        lat2, lon2 = lat_lons[i+1]
        Y.append(haversine(lon1, lat1, lon2, lat2))
gamma = 1 + len(Y) / sum([math.log(y+1) for y in Y])


column_sums = sparse_training_matrix.sum(axis=0).A1
beta = 1 + len(all_lids) / sum([math.log(column_sums[l_prime] + 1) for l_prime in all_lids])


social_dict = defaultdict(list)
for id1, id2 in social_relations:
    social_dict[id1].append(id2)


user_location_dict = {}
for user_id, location_id in training_tuples:
    if user_id in user_location_dict:
        user_location_dict[user_id].append(location_id)
    else:
        user_location_dict[user_id] = [location_id]


def is_friend(key_id, target_id):
    if target_id in social_dict.get(key_id, []):
        result = 1
    else:
        result = 0
    return result


friends_visited_locations = defaultdict(list)
for user in all_uids:
    for friend in social_dict[user]:
        friends_visited_locations[user] = list(set(friends_visited_locations[user]).union(set(user_location_dict[friend])))


eta_read = pd.read_csv(output_dir + "eta" + ".txt", sep = '\t', header = None)
eta_read.columns = ['User_ID','eta']


denominator = eta_read['eta'].sum()
numerator = 1 + len(all_uids) * len(all_lids)
eta1 = numerator/denominator
print(eta1)


csr_sparse_training_matrix = csr_matrix(sparse_training_matrix)
pl = {}

for col in range(csr_sparse_training_matrix.shape[1]):
    pl[col] = 0

for row_idx in range(csr_sparse_training_matrix.shape[0]):
    start_idx = csr_sparse_training_matrix.indptr[row_idx]
    end_idx = csr_sparse_training_matrix.indptr[row_idx + 1]
    col_indices = csr_sparse_training_matrix.indices[start_idx:end_idx]
    data = csr_sparse_training_matrix.data[start_idx:end_idx]
    for col, val in zip(col_indices, data):
        pl[col] += val


ql = defaultdict(int)

for row_idx in range(csr_sparse_training_matrix.shape[0]):
    for friend in social_dict[row_idx]:
        start_idx = csr_sparse_training_matrix.indptr[friend]
        end_idx = csr_sparse_training_matrix.indptr[friend + 1]
        col_indices = csr_sparse_training_matrix.indices[start_idx:end_idx]
        data = csr_sparse_training_matrix.data[start_idx:end_idx]
        for col, val in zip(col_indices, data):
            ql[(row_idx, col)] += val

def predict(user):
    start_time = time.time()
    Lu = user_location_dict[user]
    tc = training_check_ins[user][-1:][0][1]
    new_locations = list(set(all_lids)-set(Lu))
    Pr = np.zeros(len(new_locations))
    # Step 3.2
    for index_loc, location in enumerate (new_locations):
        p_loc = pl[location]
        q_loc = ql[user, location]
        Fpop = 1 - (p_loc + 1)**(1 - beta)
        Fsoc = 1 - (q_loc + 1)**(1 - eta1)
        Mass_loc = Fpop*Fsoc
        
        for index_li, li in enumerate(Lu):
            # Step 3.1
            x = tc - training_check_ins[user][index_li][1]
            lat1, lon1 = poi_coos[location]
            lat2, lon2 = poi_coos[li]
            y = haversine(lon1, lat1, lon2, lat2)
            Ftem = (x + 1)**(1-alpha)
            Fspa = (y + 1)**(1-gamma)
            Distance = 1/(Ftem*Fspa)
            
            # Step 3.2
            Mass_li = sparse_training_matrix[user, li]
            
            # Step 3.3
            TP = AMC.TP(li, location)
            Gravity = Mass_li * Mass_loc / Distance
            Pr[index_loc] += TP*Gravity
    top_100 = sorted(zip(new_locations, Pr), key=lambda x: x[1], reverse=True)[:100]
    
    runtime_out = open(output_dir + "runtime_" + str(top_k) + ".txt", 'a')
    runtime_out.write('\t'.join([
        str(user),
        str(time.time() - start_time),
    ]) + '\n')
    runtime_out.close()
    
    result_out = open(output_dir + "gis14_top_" + str(top_k) + ".txt", 'a')
    result_out.write('\t'.join([
        str(user),
        ','.join([str(lid) for lid, _ in top_100])
    ]) + '\n')
    result_out.close()


result_out = open(output_dir + "gis14_top_" + str(top_k) + ".txt", 'w')
runtime_out = open(output_dir + "runtime_" + str(top_k) + ".txt", 'w')

with Pool(processes=20) as pool:
    pool.map(predict, [(user) for user in all_uids])


print("waktu yang dibutuhkan untuk 1 program: ", time.time()-start_program)
