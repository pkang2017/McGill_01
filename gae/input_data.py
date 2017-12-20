import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # print adj
    return adj, features


def load_data_from_citeulike(f_in='cf-train-1-users.dat', num_v=16980, num_u=5551):  # arg: datasets
    fp = open(f_in)
    Index_I = []
    Index_J = []
    non_zero_count = 0
    Index_I_final = []
    Index_J_final = []
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        segs = map(int, segs)
        # print len(segs)
        non_zero_count += len(segs)
        Index_I.extend([i] * len(segs))
        segs = [x + 5551 for x in segs]
        Index_J.extend(segs)
        # print Index_I
        # print Index_J
        # if i == 2:
        # break

    Index_I_final = Index_I + Index_J
    Index_I_final = np.array(Index_I_final)
    # print Index_I_final

    Index_J_final = Index_J + Index_I
    Index_J_final = np.array(Index_J_final)
    # print Index_J_final

    Value = np.ones(2 * non_zero_count)

    adj = sp.csr_matrix((Value, (Index_I_final, Index_J_final)), shape=(num_u + num_v, num_u + num_v))
    return adj


def load_data_from_sub_citeulike(f_in='80_20/cf-train-1-users.dat', num_v=16980, num_u=5551):  # arg: datasets
    fp = open(f_in)
    Index_I = []
    Index_J = []
    non_zero_count = 0
    Index_I_final = []
    Index_J_final = []
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        segs = map(int, segs)
        # print len(segs)
        non_zero_count += len(segs)
        Index_I.extend([i] * len(segs))
        segs = [x + 5551 for x in segs]
        Index_J.extend(segs)
        # print Index_I
        # print Index_J
        # if i == 2:
        # break

    Index_I_final = Index_I + Index_J
    Index_I_final = np.array(Index_I_final)
    # print Index_I_final

    Index_J_final = Index_J + Index_I
    Index_J_final = np.array(Index_J_final)
    # print Index_J_final

    Value = np.ones(2 * non_zero_count)

    adj = sp.csr_matrix((Value, (Index_I_final, Index_J_final)), shape=(num_u + num_v, num_u + num_v))
    return adj



def load_data_for_mini_batch(f_in='cf-train-1-users.dat', num_v=16980, num_u=5551):  # arg: datasets
    fp = open(f_in)
    Index_I = []
    Index_J = []
    non_zero_count = 0
    Index_I_final = []
    Index_J_final = []
    for i, line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        segs = map(int, segs)
        # print len(segs)
        non_zero_count += len(segs)
        Index_I.extend([i] * len(segs))
        segs = [x for x in segs]
        Index_J.extend(segs)
        # print Index_I
        # print Index_J
        # if i == 2:
        # break

    Index_I_final = Index_I
    Index_I_final = np.array(Index_I_final)
    # print Index_I_final

    Index_J_final = Index_J
    Index_J_final = np.array(Index_J_final)
    # print Index_J_final

    Value = np.ones(non_zero_count)

    adj = sp.csr_matrix((Value, (Index_I_final, Index_J_final)), shape=(num_u, num_v))
    return adj


if __name__ == '__main__':
    adj = load_data_from_citeulike()
    adj1 = sp.coo_matrix(
        (np.array([1, 1, 1, 1, 1, 1, 0]), (np.array([0, 0, 1, 2, 3, 5, 0]), np.array([2, 3, 5, 0, 0, 1, 4]))),
        shape=(6, 6))
    print adj1.data
    print adj1
    adj1.eliminate_zeros();
    print adj1.data
    print adj1
