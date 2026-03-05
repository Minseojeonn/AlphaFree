from cmath import cos
import torch.nn.functional as F
import torch.nn as nn
import torch

class AbstractModel(nn.Module):
    '''
    Abstract Model Class
    All models should inherit this class
    and overwrite the following functions:
        - forward(self, users, items)
        - predict(self, users, items)
    Args:
        args (argparse): model parameters
        data (object): data object
    '''
    def __init__(self, args, data):
        super(AbstractModel, self).__init__()
        # basic information
        self.args = args
        self.name = args.model_name
        self.device = torch.device(args.cuda)
        self.data = data

        # basic hyper-parameters
        self.emb_dim = args.hidden_size
        self.decay = args.regs
        self.model_name = args.model_name
        self.batch_size = args.batch_size

        self.init_embedding()

    def init_embedding(self):
        '''
        Initialize the embedding, if model needs user or item embedding.
        '''
        pass


    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items

    def forward(self):
        '''
        Forward function for training
        implement this function for model
        '''
        raise NotImplementedError

    # Prediction function used when evaluation is called
    def predict(self, users, items=None):
        '''
        Prediction function for model
        Args:
            users (list): a list of user IDs
            items (list): a list of item IDs (optional), if not provided, predict scores of all items
        Returns:
            scores (numpy array): a 2-D numpy array of shape (len(users), len(items))
        '''
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # user * item
        return rate_batch.cpu().detach().numpy()
