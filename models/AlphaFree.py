from .base.model import AbstractModel
from .base.utils import csr_to_torch_sparse
import torch.nn.functional as F
import torch.nn as nn
import torch
from base.data import AlphaFreeData 

class AlphaFree(AbstractModel):
    '''AlphaFree Model'''
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau_r
        self.embed_size = args.hidden_size
        self.lm_model = args.lm_model
        self.init_item_cf_embeds = data.item_cf_embeds_original
        self.init_item_cf_embeds_aug = data.item_cf_embeds_augmented
        self.align_loss = args.lambda_align
        self.align_temperature = args.tau_a
        self.init_item_cf_embeds = torch.tensor(self.init_item_cf_embeds, dtype=torch.float32).cuda(self.device)
        self.init_item_cf_embeds_aug = torch.tensor(self.init_item_cf_embeds_aug, dtype=torch.float32).cuda(self.device)
        self.user_interact_matrix = data.user_interact_matrix
        self.aug_user_interact_matrix = data.aug_user_interact_matrix
        self.init_embed_shape = self.init_item_cf_embeds.shape[1]

        multiplier_dict = {
            'v3': 1/2,
            'llama': 9/32
        } 
        
        if(self.lm_model in multiplier_dict):
            multiplier = multiplier_dict[self.lm_model]
        else:
            multiplier = 9/32

        # MLP origin
        self.mlp_origin = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
        )
        # MLP^+ (i.e. mlp_aug)
        self.mlp_aug = nn.Sequential(
            nn.Linear(self.init_embed_shape, int(multiplier * self.init_embed_shape)),
            nn.LeakyReLU(),
            nn.Linear(int(multiplier * self.init_embed_shape), self.embed_size)
        )
    
    ## Encode user and item embeddings for original view
    def compute_origin(self, users):
        user_cf_emb = self.calculate_user_embedding(users)
        users_cf_emb = self.mlp_origin(user_cf_emb)
        items_cf_emb = self.mlp_origin(self.init_item_cf_embeds)

        return users_cf_emb, items_cf_emb

    ## Encode user and item embeddings for augmented view
    def compute_aug(self, users):
        user_cf_emb = self.calculate_user_embedding_aug(users)
        users_cf_emb = self.mlp_aug(user_cf_emb)
        items_cf_emb = self.mlp_aug(self.init_item_cf_embeds_aug)
        return users_cf_emb, items_cf_emb
    
    ## Estimate user embeddings by the augmented interaction history and item CF embeddings
    def calculate_user_embedding_aug(self, users):
        user_interaction_matrix = self.aug_user_interact_matrix[users]
        user_interaction_matrix = csr_to_torch_sparse(user_interaction_matrix).to(self.device)
        estimated_user_embeds = torch.sparse.mm(user_interaction_matrix, self.init_item_cf_embeds)
        return estimated_user_embeds
    
    ## Estimate user embeddings by the original interaction history and item CF embeddings
    def calculate_user_embedding(self, users):
        user_interaction_matrix = self.user_interact_matrix[users]
        user_interaction_matrix = csr_to_torch_sparse(user_interaction_matrix).to(self.device)
        estimated_user_embeds = torch.sparse.mm(user_interaction_matrix, self.init_item_cf_embeds)
        return estimated_user_embeds
    
    ## InfoNCE Loss for original view
    def cal_loss_origin(self, users, pos_items, neg_items):
        users_emb, all_items = self.compute_origin(users.cpu())
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        
        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        return ssm_loss, users_emb, all_items
    
    ## InfoNCE Loss for augmented view
    def cal_loss_aug(self, users, pos_items, neg_items):
        users_emb, all_items = self.compute_aug(users.cpu())
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
        return ssm_loss, users_emb, all_items
    
    ## Alignment Loss between original and augmented views
    def align_contrastive_loss(self, emb1, emb2, temperature=0.07):
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        logits = torch.matmul(emb1, emb2.T) / temperature
        labels = torch.arange(emb1.size(0)).long().to(emb1.device)

        loss_1 = F.cross_entropy(logits, labels) 
        loss_2 = F.cross_entropy(logits.T, labels)
        return (loss_1 + loss_2) / 2
    
    def forward(self, users, pos_items, neg_items):
        # InfoNCE Orign - Line 6 in Algorithm 2 (Training phase) 
        info_loss_origin, users_emb_origin, all_items_origin = self.cal_loss_origin(users, pos_items, neg_items)
        # InfoNCE Aug - Line 7 in Algorithm 2 (Training phase) 
        info_loss_aug, users_emb_aug, all_items_aug = self.cal_loss_aug(users, pos_items, neg_items)
        # Alignment loss user - Line 10 in Algorithm 2 (Training phase) 
        align_loss_u = self.align_contrastive_loss(users_emb_origin, users_emb_aug, self.align_temperature)
        # Alignment loss item - Line 10 in Algorithm 2 (Training phase)
        align_loss_i = self.align_contrastive_loss(all_items_origin[pos_items], all_items_aug[pos_items], self.align_temperature)
        # Total loss - Line 11 in Algorithm 2 (Training phase)
        return info_loss_origin + info_loss_aug + self.align_loss * (align_loss_u + align_loss_i)

    @torch.no_grad()
    def predict(self, users, items=None):
        '''
        args:
            users: 
                list of user ids
            items: 
                list of item ids, if None, predict all items
                if none, predict all items
        return:
            predicted ratings
            
        Predict the ratings of a batch of users for a batch of items.
        If items is None, predict the ratings of all items for the given users.
        '''
        if items is None:
            items = list(range(self.data.n_items))

        users_emb, all_items = self.compute_origin(users)
        
        users = users_emb.to(self.device)
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        users = F.normalize(users, dim = -1)
        items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) 

        return rate_batch.cpu().detach().numpy()