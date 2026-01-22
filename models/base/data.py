from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from .utils import helper_load, helper_load_train
from reckit import randint_choice
from scipy import sparse
import pandas as pd
import random as rd
import collections
import time
import numpy as np
import bisect
import torch

class AbstractData:
    '''
    Abstract Data Class for all models
    '''
    def __init__(self, args):
        self.path = args.data_path + args.dataset + '/cf_data/'
        self.train_file = self.path + 'train.txt'
        self.valid_file = self.path + 'valid.txt'
        self.test_file = self.path + 'test.txt'
        self.batch_size = args.batch_size
        self.neg_sample = args.neg_sample
        self.device = torch.device(args.cuda)
        self.model_name = args.model_name
        self.user_pop_max = 0
        self.item_pop_max = 0
        self.infonce = args.infonce
        self.num_workers = args.num_workers
        self.dataset = args.dataset

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []
        self.population_list = []
        self.weights = []

        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        self.test_user_list = collections.defaultdict(list)

        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        self.train_item_list = collections.defaultdict(list)
        self.Graph = None
        self.trainUser, self.trainItem, self.UserItemNet = [], [], []
        self.n_interactions = 0
        self.test_item_list = []

        #Dataloader 
        self.train_data = None
        self.train_loader = None
        self.load_data()
        self.add_special_model_attr(args)
        self.get_dataloader()

    def add_special_model_attr(self, args):
        pass

    def load_data(self):
        self.train_user_list, train_item, self.train_item_list, self.trainUser, self.trainItem = helper_load_train(
            self.train_file)
        self.valid_user_list, valid_item = helper_load(self.valid_file)
        self.test_user_list, self.test_item_list = helper_load(self.test_file)
        self.pop_dict_list = []

        temp_lst_u = [self.train_user_list, self.valid_user_list, self.test_user_list]
        temp_lst = [train_item, valid_item, self.test_item_list]
        self.users = list(set().union(*temp_lst_u))
        self.items = list(set().union(*temp_lst))
        self.items.sort()
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        print("n_users: ", self.n_users)
        print("n_items: ", self.n_items)
        
        for i in self.train_user_list:
            self.n_observations += len(self.train_user_list[i])
            self.n_interactions += len(self.train_user_list[i])
            if i in self.valid_user_list.keys():
                self.n_interactions += len(self.valid_user_list[i])
            if i in self.test_user_list.keys():
                self.n_interactions += len(self.test_user_list[i])

        # Population matrix
        pop_dict = {}
        for item, users in self.train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict.keys():
                pop_dict[item] = 1

            self.population_list.append(pop_dict[item])

        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}
        self.pop_item = pop_item
        self.pop_user = pop_user
        # Convert to a unique value.
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max        

        self.sample_items = np.array(self.items, dtype=int)

    
        self.selected_train, self.selected_valid, self.selected_test = [], [], []
        self.nu_info = []
        self.ni_info = []

        print("load finish")
      
    def get_dataloader(self):
        self.train_data = TrainDataset(self.model_name, self.users, self.train_user_list, self.user_pop_idx, self.item_pop_idx, \
                                        self.neg_sample, self.n_observations, self.n_items, self.sample_items, self.infonce, self.items, self.nu_info, self.ni_info)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

class TrainDataset(torch.utils.data.Dataset):
    '''
    Dataset Class for training
    '''
    def __init__(self, model_name, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample, \
                n_observations, n_items, sample_items, infonce, items, nu_info = None, ni_info = None):
        self.model_name = model_name
        self.users = users
        self.train_user_list = train_user_list
        self.user_pop_idx = user_pop_idx
        self.item_pop_idx = item_pop_idx
        self.neg_sample = neg_sample
        self.n_observations = n_observations
        self.n_items = n_items
        self.sample_items = sample_items
        self.infonce = infonce
        self.items = items

        self.nu_info = nu_info
        self.ni_info = ni_info
        self.cum_ni_info = np.cumsum(self.ni_info)
        self.cum_ni_info = np.insert(self.cum_ni_info, 0, 0)
        self.cum_nu_info = np.cumsum(self.nu_info)
        self.cum_nu_info = np.insert(self.cum_nu_info, 0, 0)
        
    def __getitem__(self, index):

        index = index % len(self.train_user_list)
        user = self.users[index]
        if self.train_user_list[user] == []:
            pos_items = 0
        else:
            pos_item = rd.choice(self.train_user_list[user])

        user_pop = self.user_pop_idx[user]
        pos_item_pop = self.item_pop_idx[pos_item]

        if self.infonce == 1 and self.neg_sample == -1: #in-batch
            return user, pos_item, user_pop, pos_item_pop

        elif self.infonce == 1 and self.neg_sample != -1: # InfoNCE negative sampling
            if(len(self.nu_info) > 0):
                period = bisect.bisect_right(self.cum_nu_info, index) - 1
                exclude_items = list(np.array(self.train_user_list[user]) - self.cum_ni_info[period])
                neg_items = randint_choice(self.ni_info[period], size=self.neg_sample, exclusion=exclude_items)
                neg_items = list(np.array(neg_items) + self.cum_ni_info[period])
                
   
            else:
                neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.train_user_list[user])
            neg_items_pop = self.item_pop_idx[neg_items]

            return user, pos_item, user_pop, pos_item_pop, torch.tensor(neg_items).long(), neg_items_pop

    def __len__(self):
        return self.n_observations

class AlphaFree_Data(AbstractData):
    '''
    Data Class for AlphaFree model
    Appendix D. Algorithm 1 Preprocessing phase of AlphaFree
    '''
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        self.lm_model = args.lm_model
        loading_path = args.data_path + args.dataset + '/item_info/'
        embedding_path_dict = {
            'v3': 'item_cf_embeds_large3_array.npy',
            'llama': 'item_cf_embeds_LLAMA_array.npy'
        }
        ## Load language model representations
        self.item_cf_embeds_original = np.load(loading_path + embedding_path_dict[self.lm_model])
 
        pairs = []
        for u, v in self.train_user_list.items():
            for i in v:
                pairs.append((u, i))
        pairs = pd.DataFrame(pairs, columns=['user_id', 'item_id'])
        self.topk_indices = {}    
        self.item_item_sim_dict, self.item_item_sim_matrix = self.build_item_knn_dict(pairs, args.K_c+1)
        
        ### Representation augmentation
        self.item_cf_embeds_augmented = self.item_item_sim_matrix.dot(self.item_cf_embeds_original)
        
        ### Interaction augmentation.
        row = pairs['user_id'].tolist()
        col = pairs['item_id'].tolist()
        val = len(pairs) * [1.0]
        self.user_interact_matrix = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.user_interact_matrix = normalize(self.user_interact_matrix, norm='l1', axis=1)
        self.aug_user_interact_matrix = self.augment_user_interact_matrix(pairs, self.item_item_sim_dict)
        torch.cuda.empty_cache()

    # Line 4-10 in Algorithm 1 (Preprocessing phase)
    def build_item_knn_dict(self, user_item_pairs, topk=3, batch_size=1024):
        row = user_item_pairs['user_id'].tolist()
        col = user_item_pairs['item_id'].tolist()
        val = [1.0] * len(user_item_pairs)
        A = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items))
        item_item_sim = A.T.dot(A)  # Item-Item Co-occurrence matrix
        item_item_sim = item_item_sim.tocsr()
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #load item Language Representations 
        item_embeds = torch.from_numpy(self.item_cf_embeds_original).to(device) 
        n_items = item_embeds.shape[0]
        
        sim_sum = torch.zeros(n_items, device=device)
        # Compute LR-based similarity sum for each item, in batches to save memory. - Line 7 in Algorithm 1 (Preprocessing phase)
        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            sims = torch.matmul(item_embeds[start:end], item_embeds.T)  # (batch, n_items)
            sim_sum[start:end] = sims.sum(dim=1)
        # Item-wise sum -> mean 
        embedding_based_sim_mean = (sim_sum / n_items).cpu().numpy()  
        
        # Behavioral and semantic filtering.
        item_knn = {}
        for i in range(n_items):
            row_i = item_item_sim.getrow(i)
            indices = row_i.indices
            values = row_i.data

            if len(values) == 0:
                item_knn[i] = []
                continue

            topk_idx = np.argsort(values)[-topk:][::-1]
            
            filtered_neighbors = []
            emb_i = self.item_cf_embeds_original[i]
            for j in topk_idx:
                neighbor_idx = indices[j]
                sim_val = float(values[j])
                emb_sim = float(np.dot(emb_i, self.item_cf_embeds_original[neighbor_idx]))
                if emb_sim >= embedding_based_sim_mean[i]:
                    filtered_neighbors.append((neighbor_idx, sim_val))
            
            item_knn[i] = filtered_neighbors
        
        row = []
        col = []
        value = []
        for i in item_knn:
            for j, sim in item_knn[i]:
                row.append(i)
                col.append(j)
                value.append(sim)
        item_item_sim = sparse.csr_matrix((value, (row, col)), shape=(self.n_items, self.n_items))
        degrees = np.array(item_item_sim.sum(axis=1)).ravel()  #row sum
        deg_inv = np.power(degrees, -1.0)
        deg_inv[np.isinf(deg_inv)] = 0.0
        D_inv = sparse.diags(deg_inv)
        A_hat =  D_inv @ item_item_sim
        return item_knn, A_hat

    def group_agg(self, group_data, sim_dict, key='item_id'):
        ids = group_data[key].values
        potential_int_items = set()
        for id in ids:
            temp_inter = sim_dict[id]
            for item in temp_inter:
                potential_int_items.add(item[0])
        return potential_int_items

    def augment_user_interact_matrix(self, pairs, sim_dict):
        groups = pairs.groupby('user_id')
        user_df = groups.apply(self.group_agg, sim_dict, key='item_id')
        row = []
        col = []
        val = []
        for user_id, items in user_df.items():
            for item in items:
                row.append(int(user_id))
                col.append(int(item))
                val.append(1.0)
        
        aug_matrix = sparse.csr_matrix((val, (row, col)), shape=(self.n_users, self.n_items), dtype=np.float32)
        aug_matrix = normalize(aug_matrix, norm='l1', axis=1)
        
        return aug_matrix