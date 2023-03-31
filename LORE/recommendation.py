import numpy as np
import scipy.sparse as sparse
import multiprocessing as mp
from multiprocessing import Pool
import time


from collections import defaultdict

from lib.FriendBasedCF import FriendBasedCF
from lib.KernelDensityEstimation import KernelDensityEstimation
from lib.AdditiveMarkovChain import AdditiveMarkovChain

from lib.metrics import precisionk, recallk

start_program = time.time()
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


def read_training_check_ins(training_matrix):
    check_in_data = open(check_in_file, 'r').readlines()
    training_check_ins = defaultdict(list)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not training_matrix[uid, lid] == 0:
            training_check_ins[uid].append([lid, ctime])
    return training_check_ins


def read_sparse_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num+1, poi_num+1))
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

def process_uid(cnt_uid):
    cnt, uid = cnt_uid
    if uid in ground_truth:
        start_time = time.time()
        overall_scores = [KDE.predict(uid, lid) * FCF.predict(uid, lid) * AMC.predict(uid, lid)
                            if (uid, lid) not in training_tuples else -1
                            for lid in location_ids_list]
        overall_scores = np.array(overall_scores)

        runtime_out = open(output_dir + "runtime_" + str(top_k) + ".txt", 'a')
        runtime_out.write('\t'.join([
            str(uid),
            str(time.time() - start_time),
        ]) + '\n')
        runtime_out.close()

        predicted = list(reversed(overall_scores.argsort()))[:top_k]
        actual = ground_truth[uid]
        precision.append(precisionk(actual, predicted[:10]))
        recall.append(recallk(actual, predicted[:10]))

        precrec_out = open(output_dir + "precrec_" + str(top_k) + ".txt", 'a')
        precrec_out.write('\t'.join([
            str(uid),
            str(np.mean(precision)),
            str(np.mean(recall))
        ]) + '\n')
        precrec_out.close()

        result_out = open(output_dir + "gis14_top_" + str(top_k) + ".txt", 'a')
        result_out.write('\t'.join([
            str(cnt),
            str(uid),
            ','.join([str(lid) for lid in predicted])
        ]) + '\n')
        result_out.close()
        


def main():
    # sparse_training_matrix, training_tuples = read_sparse_training_data()
    # training_check_ins = read_training_check_ins(sparse_training_matrix)
    # sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
    #                              for uid in training_check_ins}
    # social_relations = read_friend_data()
    # ground_truth = read_ground_truth()
    # poi_coos = read_poi_coos()

    # FCF.compute_friend_sim(social_relations, poi_coos, sparse_training_matrix)
    # KDE.precompute_kernel_parameters(sparse_training_matrix, poi_coos)
    # AMC.build_location_location_transition_graph(sorted_training_check_ins)

    #result_out = open("/mnt/c/Users/sarah/Downloads/cuiyue-master/cuiyue-master/RecSys -2017/6_LORE/gis14_top_" + str(top_k) + ".txt", 'w')

    # all_uids = list(range(user_num))
    # all_lids = list(range(poi_num))
    # np.random.shuffle(all_uids)

    # precision, recall = [], []
    with mp.Pool(processes=20) as pool:
        pool.map(process_uid, [(cnt, uid) for cnt, uid in enumerate(all_uids)])
    print(time.time()-start_program)
 
    # print(np.mean(precision), np.mean(recall))


if __name__ == '__main__':
    data_dir = "/home/m439162/Skripsi/Dataset/Yelp_1/"
    output_dir = "/home/m439162/Skripsi/Output/LORE/Yelp/"
    size_file = data_dir + "Yelp_data_size.txt"
    check_in_file = data_dir + "Yelp_check_ins.txt"
    train_file = data_dir + "Yelp_train.txt"
    test_file = data_dir + "Yelp_test.txt"
    social_file = data_dir + "Yelp_social_relations.txt"
    poi_file = data_dir + "Yelp_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    FCF = FriendBasedCF()
    KDE = KernelDensityEstimation()
    AMC = AdditiveMarkovChain(delta_t=3600*24, alpha=0.05)
    sparse_training_matrix, training_tuples = read_sparse_training_data()
    training_check_ins = read_training_check_ins(sparse_training_matrix)
    sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
                                 for uid in training_check_ins}
    social_relations = read_friend_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    FCF.compute_friend_sim(social_relations, poi_coos, sparse_training_matrix)
    KDE.precompute_kernel_parameters(sparse_training_matrix, poi_coos)
    AMC.build_location_location_transition_graph(sorted_training_check_ins)

    
    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    location_ids = {lid for uid, lid in training_tuples}
    location_ids_list = list(location_ids)

    precision, recall = [], []
    result_out = open(output_dir + "gis14_top_" + str(top_k) + ".txt", 'w')
    runtime_out = open(output_dir + "runtime_" + str(top_k) + ".txt", 'w')
    precrec_out = open(output_dir + "precrec_" + str(top_k) + ".txt", 'w')
    main()
