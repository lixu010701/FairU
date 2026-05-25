"""
Fairness utilities: dataset loading, sensitive attribute handling, and fairness metrics.

Supports fairness benchmark datasets (German, Bail, Credit) and
citation networks with constructed sensitive attributes.
"""

import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix


# ============================================================
# Dataset Loading
# ============================================================

def build_relationship(x, thresh=0.25):
    """Build edge list based on feature similarity (Euclidean distance)."""
    df_euclid = pd.DataFrame(
        1 / (1 + distance_matrix(x.T.T, x.T.T)),
        columns=x.T.columns, index=x.T.columns
    )
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    return np.array(idx_map)


def load_german(path="dataset/german/", label_number=1000):
    """Load German Credit dataset. Sensitive attribute: Gender."""
    dataset = "german"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("GoodCustomer")
    header.remove("OtherLoansAtStore")
    header.remove("PurposeOfLoan")

    idx_features_labels['Gender'] = idx_features_labels['Gender'].map(
        {'Female': 1, 'Male': 0}
    )

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edges_file):
        edges_unordered = np.genfromtxt(edges_file).astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(edges_file, edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["GoodCustomer"].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["Gender"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def load_bail(path="dataset/bail/", label_number=1000):
    """Load Bail/Recidivism dataset. Sensitive attribute: Race (WHITE)."""
    dataset = "bail"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("RECID")

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edges_file):
        edges_unordered = np.genfromtxt(edges_file).astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(edges_file, edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["RECID"].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["WHITE"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)



def load_bailA(path="dataset/bailA/", label_number=1000):
    """Load New Bail dataset (paper 16: Benchmark-GraphFairness). Same schema as bail, different edges."""
    dataset = "bailA"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("RECID")

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    edges_unordered = np.genfromtxt(edges_file).astype('int')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["RECID"].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["WHITE"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def load_credit(path="dataset/credit/", label_number=1000):
    """Load Credit Defaulter dataset. Sensitive attribute: Age."""
    dataset = "credit"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("NoDefaultNextMonth")
    header.remove("Single")

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edges_file):
        edges_unordered = np.genfromtxt(edges_file).astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(edges_file, edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["NoDefaultNextMonth"].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["Age"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)



def load_creditA(path="dataset/creditA/", label_number=1000):
    """Load New Credit dataset (paper 16)."""
    dataset = "creditA"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("NoDefaultNextMonth")
    header.remove("Single")
    
    

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    edges_unordered = np.genfromtxt(edges_file).astype('int')

    
    
    

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["NoDefaultNextMonth"].values
    

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["Age"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)



def load_germanA(path="dataset/germanA/", label_number=1000):
    """Load New German dataset (paper 16)."""
    dataset = "germanA"
    idx_features_labels = pd.read_csv(os.path.join(path, f"{dataset}.csv"))
    header = list(idx_features_labels.columns)
    header.remove("GoodCustomer")
    
    header.remove("PurposeOfLoan")
    header.remove("OtherLoansAtStore")

    edges_file = os.path.join(path, f"{dataset}_edges.txt")
    edges_unordered = np.genfromtxt(edges_file).astype('int')

    # Gender mapping for german/germanA
    idx_features_labels.loc[idx_features_labels["Gender"]=="Female", "Gender"] = 1
    idx_features_labels.loc[idx_features_labels["Gender"]=="Male", "Gender"] = 0

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["GoodCustomer"].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["Gender"].values.astype(int)
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)



def load_syn(name='syn1', path=None, train_ratio=0.6, seed=20):
    """Load Syn-1/Syn-2 synthetic datasets from paper 16.

    Files: {name}_edges.txt (comma-sep, directed edges), {name}_feat.csv,
           {name}_label.txt (per-line int), {name}_sens.txt (per-line int).
    """
    import random
    if path is None:
        path = f"dataset/{name}/"
    labels = np.loadtxt(os.path.join(path, f"{name}_label.txt"), dtype=int)
    sens = np.loadtxt(os.path.join(path, f"{name}_sens.txt"), dtype=int)
    features = np.loadtxt(os.path.join(path, f"{name}_feat.csv"), delimiter=',')
    edges = np.loadtxt(os.path.join(path, f"{name}_edges.txt"), delimiter=',', dtype=int)

    n = features.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    # make symmetric
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(n)

    features = torch.FloatTensor(np.array(sp.csr_matrix(features, dtype=np.float32).todense()))
    labels = torch.LongTensor(labels)

    np.random.seed(seed)
    random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx_train = idx[:int(n * train_ratio)]
    idx_val = idx[int(n * train_ratio):int(n * (1 + train_ratio) / 2)]
    idx_test = idx[int(n * (1 + train_ratio) / 2):]

    sens = torch.LongTensor(sens)
    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def load_syn1(path="dataset/syn1/", label_number=1000):
    return load_syn(name='syn1', path=path)


def load_syn2(path="dataset/syn2/", label_number=1000):
    return load_syn(name='syn2', path=path)



def _load_twitter(name, path, sens_col, label_col, train_ratio=0.6, seed=20):
    """Shared loader for paper 16 Twitter datasets (sport, occupation)."""
    import random
    df = pd.read_csv(os.path.join(path, f"{name}.csv"))
    df = df.drop_duplicates(subset="user_id").reset_index(drop=True)

    header = list(df.columns)
    for col in ["user_id", "embeddings", label_col, sens_col]:
        header.remove(col)

    features = sp.csr_matrix(df[header].values, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(df[label_col].values.astype(int))
    sens = torch.LongTensor(df[sens_col].values.astype(int))

    user_ids = df["user_id"].values.astype("int64")
    idx_map = {int(j): i for i, j in enumerate(user_ids)}
    n = len(user_ids)

    edges_raw = np.loadtxt(os.path.join(path, f"{name}_edges.txt"), dtype="int64")
    edges_mapped = []
    for u, v in edges_raw:
        u = int(u); v = int(v)
        if u in idx_map and v in idx_map:
            edges_mapped.append([idx_map[u], idx_map[v]])
    edges = np.asarray(edges_mapped, dtype=int)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(n)

    random.seed(seed)
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    idx_train = idx[:int(n * train_ratio)]
    idx_val = idx[int(n * train_ratio):int(n * (1 + train_ratio) / 2)]
    idx_test = idx[int(n * (1 + train_ratio) / 2):]

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def load_sport(path="dataset/sport/", label_number=3508):
    return _load_twitter("sport", path, sens_col="race", label_col="sport")


def load_occupation(path="dataset/occupation/", label_number=6951):
    return _load_twitter("occupation", path, sens_col="gender", label_col="area")


def load_pokec(dataset_name, path="dataset/pokec_z/", label_number=500):
    """Load Pokec-z or Pokec-n dataset. Sensitive attribute: Region."""
    if 'pokec_n' in dataset_name:
        csv_file = os.path.join(path, "region_job_2.csv")
        rel_file = os.path.join(path, "region_job_2_relationship.txt")
    else:
        csv_file = os.path.join(path, "region_job.csv")
        rel_file = os.path.join(path, "region_job_relationship.txt")

    idx_features_labels = pd.read_csv(csv_file)
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove("region")
    header.remove("I_am_working_in_field")

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["I_am_working_in_field"].values
    labels[labels > 1] = 1

    # Build adjacency from relationship file (with ID mapping)
    idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(rel_file, dtype=np.int64)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
    ).reshape(edges_unordered.shape)
    # Remove edges with unmapped nodes
    valid = ~np.isnan(edges).any(axis=1) if edges.dtype == float else np.ones(len(edges), dtype=bool)
    edges = edges[valid]
    n_nodes = features.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_nodes, n_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):]
    )

    sens = idx_features_labels["region"].values.astype(int)
    sens[sens > 0] = 1
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def load_nba(path="dataset/nba/", label_number=100):
    """Load NBA dataset. Sensitive attribute: Country (US vs non-US)."""
    idx_features_labels = pd.read_csv(os.path.join(path, "nba.csv"))
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove("country")
    header.remove("SALARY")

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels["SALARY"].values
    labels[labels > 1] = 1

    # Build adjacency with ID mapping
    idx = np.array(idx_features_labels["user_id"], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "nba_relationship.txt"), dtype=np.int64)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64
    ).reshape(edges_unordered.shape)
    n_nodes = features.shape[0]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_nodes, n_nodes), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.2 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.2 * len(label_idx_1)), label_number // 2)]
    )
    idx_val = np.append(
        label_idx_0[int(0.2 * len(label_idx_0)):int(0.55 * len(label_idx_0))],
        label_idx_1[int(0.2 * len(label_idx_1)):int(0.55 * len(label_idx_1))]
    )
    idx_test = np.append(
        label_idx_0[int(0.55 * len(label_idx_0)):],
        label_idx_1[int(0.55 * len(label_idx_1)):]
    )

    sens = idx_features_labels["country"].values.astype(int)
    sens[sens > 0] = 1
    sens = torch.LongTensor(sens)

    return (adj, features, labels,
            torch.LongTensor(idx_train), torch.LongTensor(idx_val),
            torch.LongTensor(idx_test), sens)


def feature_norm(features):
    """Normalize features to [-1, 1] range per feature."""
    min_values = features.min(dim=0)[0]
    max_values = features.max(dim=0)[0]
    width = (max_values - min_values).clamp(min=1e-8)
    return 2 * (features - min_values) / width - 1


def load_dataset(name, data_root="dataset/"):
    """
    Unified dataset loader.

    Parameters
    ----------
    name : str
        Dataset name: 'german', 'bail', 'credit'.
    data_root : str
        Root directory containing dataset subfolders.

    Returns
    -------
    tuple : (adj, features, labels, idx_train, idx_val, idx_test, sensitive)
        - adj: scipy sparse adjacency matrix
        - features: torch.FloatTensor [N, F]
        - labels: torch.LongTensor [N]
        - idx_train/val/test: torch.LongTensor
        - sensitive: torch.LongTensor [N] (binary)
    """
    loaders = {
        'german': load_german,
        'germanA': load_germanA,
        'bail': load_bail,
        'sport': load_sport,
        'occupation': load_occupation,
        'syn1': load_syn1,
        'syn2': load_syn2,
        'bailA': load_bailA,
        'credit': load_credit,
        'creditA': load_creditA,
        'pokec_z': load_pokec,
        'pokec_n': load_pokec,
        'nba': load_nba,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    if name in ('pokec_z', 'pokec_n'):
        return loaders[name](name, path=os.path.join(data_root, name, ""))
    return loaders[name](path=os.path.join(data_root, name, ""))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


# ============================================================
# Sensitive Attribute Construction (for datasets without explicit sensitive attrs)
# ============================================================

def construct_sensitive_by_degree(adj, method='median'):
    """
    Construct binary sensitive attribute based on node degree.
    High-degree nodes → group 1 (structural privilege).
    Low-degree nodes → group 0.
    """
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            degrees = torch.sparse.sum(adj, dim=1).to_dense()
        else:
            degrees = adj.sum(dim=1)
    else:
        degrees = torch.FloatTensor(np.array(adj.sum(axis=1)).flatten())

    if method == 'median':
        threshold = degrees.median()
    elif method == 'mean':
        threshold = degrees.mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    return (degrees > threshold).long()


# ============================================================
# Fairness Metrics (aligned with DAB-GNN AAAI 2025 standard)
# ============================================================

from sklearn.metrics import roc_auc_score, f1_score


def fair_metric(pred, labels, sens):
    """
    Compute ΔSP and ΔEO (following DAB-GNN implementation).

    Parameters
    ----------
    pred : np.ndarray
        Binary predictions.
    labels : np.ndarray
        Ground truth labels.
    sens : np.ndarray
        Binary sensitive attributes.

    Returns
    -------
    parity : float
        ΔSP = |P(Ŷ=1|S=0) - P(Ŷ=1|S=1)|
    equality : float
        ΔEO = |P(Ŷ=1|Y=1,S=0) - P(Ŷ=1|Y=1,S=1)|
    """
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)

    parity = abs(sum(pred[idx_s0]) / max(sum(idx_s0), 1) -
                 sum(pred[idx_s1]) / max(sum(idx_s1), 1))
    equality = abs(sum(pred[idx_s0_y1]) / max(sum(idx_s0_y1), 1) -
                   sum(pred[idx_s1_y1]) / max(sum(idx_s1_y1), 1))

    return float(parity), float(equality)


def compute_fairness_metrics(output, labels, sensitive, mask=None):
    """
    Compute all metrics: Acc%, AUC%, F1%, ΔSP%, ΔEO%.
    Aligned with DAB-GNN (AAAI 2025) evaluation standard.

    All values returned as percentages (0-100).

    Parameters
    ----------
    output : torch.Tensor
        Model output logits [N, C] or [N].
    labels : torch.Tensor
        Ground truth [N].
    sensitive : torch.Tensor
        Sensitive attributes [N].
    mask : torch.Tensor, optional
        Boolean mask for evaluation subset.

    Returns
    -------
    dict with keys: acc, auc, f1, sp, eo (all in percentage)
    """
    if mask is not None:
        output = output[mask]
        labels = labels[mask]
        sensitive = sensitive[mask]

    labels_np = labels.cpu().numpy()
    sens_np = sensitive.cpu().numpy()

    if output.dim() == 1:
        # Single logit (binary)
        preds = (output > 0).long()
        output_for_auc = output.detach().cpu().numpy()
    else:
        # Multi-class logits
        preds = output.argmax(dim=1)
        # For AUC: use probability of class 1
        if output.shape[1] == 2:
            output_for_auc = output[:, 1].detach().cpu().numpy()
        else:
            output_for_auc = output.detach().cpu().numpy()

    preds_np = preds.cpu().numpy()

    # Accuracy
    acc = float(np.mean(preds_np == labels_np)) * 100.0

    # AUC-ROC
    try:
        auc = roc_auc_score(labels_np, output_for_auc) * 100.0
    except ValueError:
        auc = 50.0  # fallback for single-class predictions

    # F1 Score (macro: average across classes, penalizes majority-class-only predictions)
    f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0) * 100.0

    # Fairness: ΔSP and ΔEO
    sp, eo = fair_metric(preds_np, labels_np, sens_np)
    sp *= 100.0
    eo *= 100.0

    return {
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'sp': sp,
        'eo': eo,
    }
