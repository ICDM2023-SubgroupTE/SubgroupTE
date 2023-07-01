import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
SEED = 1111
np.random.seed(SEED)
    
def Load_split_dataset(config):
    params = config["data_loader"]
    if params['data'] == 'OUD':
        data = load_OUD(config)
        n_samples = len(data)
    else:
        if params['data'] == 'IHDP':
            data = create_IHDP()
        else:
            data = create_synth(params["n_samples"])
        n_samples, n_feat = data[0].shape 
        config["hyper_params"]["input_dim"] = n_feat
        
    n_train = int(n_samples*params["train_ratio"])
    n_test = int(n_samples*params["test_ratio"])

    index = np.random.RandomState(seed=SEED).permutation(n_samples)
    train_index = index[:n_train]
    test_index = index[n_train:n_train+n_test]
    valid_index = index[n_train+n_test:]
    
    if params['data'] == 'OUD':        
        convert_numeric = Convert_to_numeric()
        train_set, convert_numeric = OUD_dataset(data, train_index, convert_numeric, is_train=True)
        valid_set, _ = OUD_dataset(data, valid_index, convert_numeric)
        test_set, _ = OUD_dataset(data, test_index, convert_numeric)
        config["hyper_params"]["input_dim"] = convert_numeric.feat_idx

    else:
        train_set = synth_dataset(data, train_index)
        valid_set = synth_dataset(data, valid_index)
        test_set = synth_dataset(data, test_index)
    
    config["data_loader"]["n_samples"] = n_samples
    config["data_loader"]["n_train"] = len(train_index)
    config["data_loader"]["n_valid"] = len(valid_index)
    config["data_loader"]["n_test"] = len(test_index)
    
    return config, train_set, valid_set, test_set


def synth_dataset(data, index):
    (X, T, Y, Y_0, Y_1, Y_cf, TE) = data
    dataset = {
        'X': X[index],
        'T': T[index],
        'Y': Y[index],
        'TE': TE[index]
    }
    return dataset


def OUD_dataset(data, index, convert_numeric, is_train=False):
    data = data.loc[index,]
    X = convert_numeric.forward(data['X'], is_train=is_train)
    
    dataset = {
        'X': X,
        'T': np.array(data['T']).reshape(-1,1),
        'Y': np.array(data['Y']).reshape(-1,1),
        'Age': np.array(data['Age']).reshape(-1,1),
        'Gender': np.array(data['Gender']).reshape(-1,1)
    }
     
    return dataset, convert_numeric



#######################################################################################################################
# OUD data
#######################################################################################################################
def load_OUD(config):
    drug_names = ['input_1103640.pkl','input_1133201.pkl','input_1714319.pkl']
    dataset = pd.DataFrame()
    for drug in drug_names:
        path = os.path.join(config['path'], drug)
        data = pd.DataFrame(pickle.load(open(path, 'rb')), columns=['patient','features','Y'])
        if config['data_loader']['target_drug'] in drug:
            data['T'] = 1
        else:
            data['T'] = 0
        dataset = pd.concat([dataset, data])
    
    dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True).to_dict('series')
    
    n_samples = len(dataset['patient'])
    dataset['X'] = [pd.NA for _ in range(n_samples)]
    dataset['Age'] = [pd.NA for _ in range(n_samples)]
    dataset['Gender'] = [pd.NA for _ in range(n_samples)]
    dataset['n_visit'] = [pd.NA for _ in range(n_samples)]
    for idx in range(n_samples):
        dataset['X'][idx] = dataset['features'][idx][1]
        dataset['Age'][idx] = dataset['features'][idx][2]
        dataset['Gender'][idx] = dataset['features'][idx][3]
        dataset['n_visit'][idx] = len(dataset['X'][idx])
        
    dataset = pd.DataFrame(dataset)    
    dataset = dataset[dataset.n_visit<=200].reset_index(drop=True)
    
    return dataset
    
    
class Convert_to_numeric:
    def __init__(self,):
        super(Convert_to_numeric, self).__init__()
        self.feat_idx = 0
        self.feat_dict = {}
    
    def transform(self, seqs, is_train):
        code_list = list()
        for sub_seq in seqs:
            sub_list=list()
            for code in sub_seq:
                if is_train and code not in self.feat_dict.keys():
                    self.feat_dict[code] = self.feat_idx
                    self.feat_idx += 1
                if code in self.feat_dict.keys():
                    sub_list.append(self.feat_dict[code])
            code_list.append(sub_list)
        return code_list

    def forward(self, input, is_train=False):
        n_samples = len(input)
        X = [pd.NA for _ in range(n_samples)]
        for idx in range(n_samples):
            X[idx] = self.transform(input.iloc[idx], is_train)
        return X

#######################################################################################################################
# Synthetic data
#######################################################################################################################

def create_synth(n_samples, SEED=1111):
    np.random.seed(seed=SEED)
    X = np.round(np.random.normal(size=(n_samples, 1), loc=66.0, scale=4.1))  # age
    X = np.block([X, np.round(
        np.random.normal(size=(n_samples, 1), loc=6.2, scale=1.0) * 10.0) / 10.0])  # white blood cell count
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=0.8, scale=0.1) * 10.0) / 10.0])  # Lymphocyte count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=183.0, scale=20.4))])  # Platelet count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=68.0, scale=6.6))])  # Serum creatinine
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=31.0, scale=5.1))])  # Aspartete aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=26.0, scale=5.1))])  # Alanine aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=339.0, scale=51))])  # Lactate dehydrogenase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=76.0, scale=21))])  # Creatine kinase
    X = np.block([X, np.floor(np.random.uniform(size=(n_samples, 1)) * 11) + 4])  # Time from study 4~14
    TIME = X[:, 9]

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    T = np.random.binomial(1, 0.5, size=(n_samples,1))

    # sample random coefficients
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB = np.random.choice(coeffs_, size=9, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    MU_0 = np.dot(X[:, 0:9], BetaB)
    MU_1 = np.dot(X[:, 0:9], BetaB)

    logi0 = lambda x: 1 / (1 + np.exp(-(x - 9))) + 5
    logi1 = lambda x: 5 / (1 + np.exp(-(x - 9)))

    MU_0 = MU_0 + logi0(TIME)
    MU_1 = MU_1 + logi1(TIME)

    Y_0 = (np.random.normal(scale=0.1, size=len(X)) + MU_0).reshape(-1,1)
    Y_1 = (np.random.normal(scale=0.1, size=len(X)) + MU_1).reshape(-1,1)

    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1
    
    TE = Y_1 - Y_0

    return (X, T, Y, Y_0, Y_1, Y_cf, TE)


def normalize_mean(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (result[feature_name] - result[feature_name].mean()) / result[feature_name].std()
    return result

#######################################################################################################################
# IHDP data
#######################################################################################################################

def create_IHDP(noise=0.1):
    Dataset= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)

    col = ["Treatment", "Response", "Y_CF", "mu0", "mu1", ]

    for i in range(1, 26):
        col.append("X" + str(i))
    Dataset.columns = col
    Dataset.head()

    num_samples = len(Dataset)

    feat_name = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'

    X = np.array(Dataset[feat_name.split()])
    T = np.array(Dataset['Treatment']).reshape(-1,1)

    Y_0 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu0']).reshape(-1,1)
    Y_1 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu1']).reshape(-1,1)

    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1

    TE = Y_1 - Y_0

    return (X, T, Y, Y_0, Y_1, Y_cf, TE)
    