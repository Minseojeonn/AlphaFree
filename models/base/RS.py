from .util.cython.tools import float_type, is_ndarray
from .util import argmax_top_k
from .data import AlphaFree_Data
from .evaluator.proxy_evaluator import ProxyEvaluator
from .util import DataIterator
from .utils import *
from .util.gdown import download_from_gdrive
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import datetime
import torch
import json
import time
import os


class AbstractRS(nn.Module):
    '''
    Abstract training class for recommender system
    '''
    def __init__(self, args, special_args) -> None:
        super(AbstractRS, self).__init__()
        self.args = args
        self.special_args = special_args
        self.device = torch.device(args.cuda)
        self.test_only = args.test_only

        self.Ks = args.Ks
        self.patience = args.patience
        self.model_name = args.model_name
        self.neg_sample = args.neg_sample
        self.inbatch = self.args.infonce == 1 and self.args.neg_sample == -1
    
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.verbose = args.verbose

        self.dataset_name = args.dataset

        self.data = AlphaFree_Data(args)
        
        self.n_users = self.data.n_users
        self.n_items = self.data.n_items
        self.train_user_list = self.data.train_user_list
        self.valid_user_list = self.data.valid_user_list
     
        self.user_pop = torch.tensor(self.data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(self.data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = self.data.user_pop_max
        self.item_pop_max = self.data.item_pop_max 

        self.running_model = args.model_name + '_batch' if self.inbatch else args.model_name
        exec('from models.'+ args.model_name + ' import ' + self.running_model) # import the model first
        self.model = eval(self.running_model + '(args, self.data)') # initialize the model with the graph
        self.model.cuda(self.device)
        self.preperation_for_saving(args, special_args)
        self.evaluators, self.eval_names = self.get_evaluators(self.data) # load the evaluators

    # Function inference process using pre-trained weights
    def inference_only(self):
        self.model, self.start_epoch = self.restore_checkpoint_using_pretrain(self.model, self.dataset_name, self.device) # restore the checkpoint
        self.model.eval() # evaluate the best model

        n_rets = {}
        for i,evaluator in enumerate(self.evaluators[:]):
            _, __, n_ret = evaluation(self.args, self.data, self.model, self.start_epoch, self.base_path, evaluator, self.eval_names[i])
            n_rets[self.eval_names[i]] = n_ret

        self.recommend_top_k()

    # Function training process from scratch
    def execute(self):
        self.save_args() 
        perf_str = str(self.args)
        with open(self.base_path + 'stats.txt','a') as f:
            f.write(perf_str+"\n")

        self.model, self.start_epoch = self.restore_checkpoint(self.model, self.base_path, self.device) # restore the checkpoint
        
        start_time = time.time()
        # train the model if not test only
        if not self.test_only:
            print("start training") 
            self.train()
            # test the model
            print("start testing")
            self.model = self.restore_best_checkpoint(self.data.best_valid_epoch, self.model, self.base_path, self.device)
        end_time = time.time()
        self.model.eval() # evaluate the best model
        print_str = "The best epoch is % d, total training cost is %.1f" % (max(self.data.best_valid_epoch, self.start_epoch), end_time - start_time)
        with open(self.base_path +'stats.txt', 'a') as f:
            f.write(print_str + "\n")

        n_rets = {}
        for i,evaluator in enumerate(self.evaluators[:]):
            _, __, n_ret = evaluation(self.args, self.data, self.model, self.data.best_valid_epoch, self.base_path, evaluator, self.eval_names[i])
            n_rets[self.eval_names[i]] = n_ret

        self.recommend_top_k()
        

    def save_args(self):
        with open(self.base_path + '/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

   
    def train(self) -> None:
        self.set_optimizer() # get the optimizer
        self.flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if self.flag: # early stop
                break
            # All models
            t1=time.time()
            losses = self.train_one_epoch(epoch) # train one epoch
            t2=time.time()
            self.document_running_loss(losses, epoch, t2-t1) # report the loss
            if (epoch + 1) % self.verbose == 0: # evaluate the model
                self.eval_and_check_early_stop(epoch)
        visualize_and_save_log(self.base_path +'stats.txt', self.dataset_name)

    def train_one_epoch(self, epoch):
        raise NotImplementedError
    
    def preperation_for_saving(self, args, special_args):
        self.formatted_today=datetime.datetime.now().strftime('%m%d_%H%M%S') + '_'

        if(self.test_only == False):
            prefix = self.formatted_today + args.saveID
        else:
            prefix = args.saveID
        self.saveID = prefix + '_'  + "_Ks=" + str(args.Ks) + '_patience=' + str(args.patience)\
            +"_batch_size=" + str(args.batch_size)\
                + "_neg_sample=" + str(args.neg_sample) + "_lr=" + str(args.lr) + "_seed=" + str(args.seed)

        for arg in special_args:
            print(arg, getattr(args, arg))
            self.saveID += "_" + arg + "=" + str(getattr(args, arg))
    
        
        self.modify_saveID()

        self.base_path = './weights/{}/{}/{}'.format(self.dataset_name, self.running_model, self.saveID)
        self.checkpoint_buffer=[]
        ensureDir(self.base_path)

    def modify_saveID(self):
        pass

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)

    def document_running_loss(self, losses:list, epoch, t_one_epoch, prefix=""):
        loss_str = ', '.join(['%.5f']*len(losses)) % tuple(losses)
        perf_str = prefix + 'Epoch %d [%.1fs]: train==[' % (
                epoch, t_one_epoch) + loss_str + ']'
        with open(self.base_path + 'stats.txt','a') as f:
                f.write(perf_str+"\n")
    
    def document_hyper_params_results(self, base_path, n_rets):
        overall_path = '/'.join(base_path.split('/')[:-1]) + '/'
        hyper_params_results_path = overall_path + self.formatted_today + self.dataset_name + '_' + self.model_name + '_' + self.args.saveID + '_hyper_params_results.csv'

        results = {'notation': self.formatted_today, 'best_epoch': max(self.data.best_valid_epoch, self.start_epoch), 'max_epoch': self.max_epoch, 'Ks': self.Ks, 'n_layers': self.n_layers, 'batch_size': self.batch_size, 'neg_sample': self.neg_sample, 'lr': self.lr}
        for special_arg in self.special_args:
            results[special_arg] = getattr(self.args, special_arg)

        for k, v in n_rets.items():
            if('test_id' not in k):
                for metric in ['recall', 'ndcg', 'hit_ratio']:
                    results[k + '_' + metric] = round(v[metric], 4)
        frame_columns = list(results.keys())
        if os.path.exists(hyper_params_results_path):
            hyper_params_results = pd.read_csv(hyper_params_results_path)
        else:
            hyper_params_results = pd.DataFrame(columns=frame_columns)

        hyper_params_results = hyper_params_results._append(results, ignore_index=True)
   
        hyper_params_results.to_csv(hyper_params_results_path, index=False, float_format='%.4f')
   

    def recommend_top_k(self):
        test_users = list(self.data.test_user_list.keys())
        
        eval_train_user_list = self.data.train_user_list
        dump_dict = merge_user_list([eval_train_user_list,self.data.valid_user_list])
        recommended_top_k = {}
        recommended_scores = {}
        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        for batch_id, batch_users in enumerate(test_users):
            ranking_score = self.model.predict(batch_users, None)  # (B,N)
            if not is_ndarray(ranking_score, float_type):
                ranking_score = np.array(ranking_score, dtype=float_type)
            
            for idx, user in enumerate(batch_users):
                dump_items = dump_dict[user]
                dump_items = [ x for x in dump_items if not x in self.data.test_user_list[user] ]
                ranking_score[idx][dump_items] = -np.inf

                recommended_top_k[user] = argmax_top_k(ranking_score[idx], self.Ks)
                recommended_scores[user] = ranking_score[idx][recommended_top_k[user]]

        #rank score
        with open(self.base_path + '/recommend_top_k.txt', 'w') as f:
            for u, v in recommended_top_k.items():
                f.write(str(int(u)))
                for i in range(self.Ks):
                    f.write(' ' + str(int(v[i])))
                f.write('\n')
        with open(self.base_path + '/recommend_top_k_with_score.txt', 'w') as f:
            for u, v in recommended_top_k.items():
                f.write(str(int(u)))
                for i in range(self.Ks):
                    f.write(' ' + str(int(v[i])) + '+' + str(round(recommended_scores[u][i], 4)))
                f.write('\n')
        print('finish recommend top k')
    
    # define the evaluation process
    def eval_and_check_early_stop(self, epoch):
        self.model.eval()

        for i,evaluator in enumerate(self.evaluators):
            tt1 = time.time()
            is_best, temp_flag, n_ret = evaluation(self.args, self.data, self.model, epoch, self.base_path, evaluator, self.eval_names[i])
            tt2 = time.time()
            print("Evaluating %d [%.1fs]: %s" % (i, tt2 - tt1, self.eval_names[i]))
            if is_best:
                checkpoint_buffer=save_checkpoint(self.model, epoch, self.base_path, self.checkpoint_buffer, self.args.max2keep)
            
            # early stop
            if temp_flag:
                self.flag = True
        
        
        self.model.train()
    
    # load the checkpoint
    def restore_checkpoint(self, model, checkpoint_dir, device, force=False, pretrain=False):
        """
        If a checkpoint exists, restores the PyTorch model from the checkpoint.
        Returns the model and the current epoch.
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        if not cp_files:
            print('No saved model parameters found')
            if force:
                raise Exception("Checkpoint not found")
            else:
                return model, 0,

        epoch_list = []

        regex = re.compile(r'\d+')

        for cp in cp_files:
            epoch_list.append([int(x) for x in regex.findall(cp)][0])

        epoch = max(epoch_list)

        if not force:
            print("Which epoch to load from? Choose in range [0, {})."
                .format(epoch), "Enter 0 to train from scratch.")
            print(">> ", end = '')

            if self.args.clear_checkpoints:
                print("Clear checkpoint")
                clear_checkpoint(checkpoint_dir)
                return model, 0,

            inp_epoch = epoch
            if inp_epoch not in range(epoch + 1):
                raise Exception("Invalid epoch number")
            if inp_epoch == 0:
                print("Checkpoint not loaded")
                clear_checkpoint(checkpoint_dir)
                return model, 0,
        else:
            print("Which epoch to load from? Choose in range [0, {}).".format(epoch))
            inp_epoch = int(input())
            if inp_epoch not in range(0, epoch):
                raise Exception("Invalid epoch number")

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))

        try:
            if pretrain:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("=> Successfully restored checkpoint (trained for {} epochs)"
                .format(checkpoint['epoch']))
        except:
            print("=> Checkpoint not successfully restored")
            raise

        return model, inp_epoch
    
    def restore_best_checkpoint(self, epoch, model, checkpoint_dir, device):
        """
        Restore the best performance checkpoint
        """
        cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
                    if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

        filename = os.path.join(checkpoint_dir,
                                'epoch={}.checkpoint.pth.tar'.format(epoch))

        print("Loading from checkpoint {}?".format(filename))

        checkpoint = torch.load(filename, map_location = str(device))

        model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))

        return model
    
    def restore_checkpoint_using_pretrain(self, model, dataset, device):
        """
        Restore pre-trained checkpoint.
        Download pre-trained weights using google drive download.
        """
        
        file_name = './pretrained/' + dataset + '.pth.tar'
        
        if os.path.isfile(file_name) == False:
            pretrain_file_id_dict = {
                "amazon_movie": "1a-4vro-yS-trNguL7lnpzFvntHjCO3xj",
                "amazon_book_2014": "1bXw1u6Yq8Tbv2rX5kz1k1K5b3F6ZlV3D",
                "amazon_video": "1cYq3F2G7H8J9K0L1M2N3O4P5Q6R7S8T9",
                "amazon_baby": "1dZx5Y6W7X8Y9Z0A1B2C3D4E5F6G7H8I9",
                "steam": "1eA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P",
                "amazon_beauty_personal": "1fP1Q2R3S4T5U6V7W8X9Y0Z1A2B3C4D5E",
                "amazon_health": "1gF6G7H8I9J0K1L2M3N4O5P6Q7R8S9T0U" 
            }
            download_from_gdrive("", pretrain_file_id_dict, file_name)
        
        checkpoint = torch.load(file_name, map_location = str(device))
        model.load_state_dict(checkpoint['state_dict'])
        print("Successfully restored pre-trained checkpoint.") 

        return model
    
    def get_evaluators(self, data):
        K_value = self.args.Ks
       
        eval_train_user_list = data.train_user_list

        interaction_list_view = None
        eval_valid = ProxyEvaluator(data,eval_train_user_list,data.valid_user_list,top_k=[K_value],group_view=interaction_list_view, dump_dict=merge_user_list([eval_train_user_list, data.test_user_list]))  
        eval_test = ProxyEvaluator(data,eval_train_user_list,data.test_user_list,top_k=[K_value],group_view=interaction_list_view, dump_dict=merge_user_list([eval_train_user_list, data.valid_user_list]))
        evaluators=[eval_valid, eval_test]
        eval_names=["valid", "test"]

        return evaluators, eval_names

class AlphaFreeRS(AbstractRS):
    '''
    AlphaFree training class 
    '''
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
    
    def save_final_model(self, model, path):
        torch.save(model.state_dict(), path + '/model.pt')
        print("Model saved at", path + '/model.pt')
    
    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, _, _, neg_items = batch[0], batch[1], batch[2], batch[3], batch[4]
            self.model.train()
            loss = self.model(users, pos_items, neg_items)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches]