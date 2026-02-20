from .base.utils import csr_to_torch_sparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class AlphaFreeRecDemo(torch.nn.Module):
    """
        Demo class for inference with AlphaFree model
        This class only uses the original view for inference.
        
        This class only supports the amazon_movie dataset, with a hard-coded number of items.
        For other datasets, please modify the number of items and the loading of item_cf_embeds accordingly.
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.tau = args.tau_r
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.n_items = 25612 # default for amazon_movie
        self.device = args.cuda
        self.init_item_cf_embeds = np.load(args.data_path + args.dataset + '/item_info/item_cf_embeds_llama_array.npy', allow_pickle=True)
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]
        multiplier = 1/2
        
        # MLP origin
        self.mlp_origin = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
        )

    def calculate_interaction_embedding(self, interacted_items):
        '''
        args:
            interacted_items: 
                list of items
        return:
            interaction_emb: 
                estimated embedding
        '''
        interacted_items = self.init_item_cf_embeds[interacted_items]
        user_cf_emb = torch.mean(interacted_items, dim=0, keepdim=True)
        return user_cf_emb
        
    def compute_origin(self, interactions):
        user_cf_emb = self.calculate_interaction_embedding(interactions).cuda(self.device)
        users_cf_emb = self.mlp_origin(user_cf_emb)
        items_cf_emb = self.mlp_origin(self.init_item_cf_embeds)

        return users_cf_emb, items_cf_emb
    
    @torch.no_grad()
    def predict(self, interactions, items=None):
        '''
        args:
            interactions: 
                list interacted items
            items: 
                list of item ids, if None, predict all items
                if none, predict all items
        return:
            predicted ratings
            
        Predict the ratings of a batch of users for a batch of items.
        If items is None, predict the ratings of all items for the given users.
        '''
        if items is None:
            items = list(range(self.n_items))
        users_emb, all_items = self.compute_origin(interactions)
        users = users_emb.to(self.device)
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        users = F.normalize(users, dim = -1)
        items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) 

        return torch.topk(rate_batch, k=20)[1]
