from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import collections
import numpy as np
import random
import torch
import math
import os
import re

# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

    return user_dict_list, item_dict,

def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = defaultdict(list)
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(int, line.split()))
            user, items = parts[0], parts[1:]
            if not items:
                continue

            item_dict.update(items)
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            user_dict_list[user] = items

            for item in items:
                item_dict_list[item].append(user)

    return user_dict_list, item_dict, dict(item_dict_list), trainUser, trainItem

def merge_user_list(user_lists):
    out = collections.defaultdict(list)
    # Loop over each user list
    for user_list in user_lists:
        # Loop over each user in the user list
        for key, item in user_list.items():
            out[key] = out[key] + item
    return out


def merge_user_list_no_dup(user_lists):
    out = collections.defaultdict(list)
    for user_list in user_lists:
        for key, item in user_list.items():
            out[key] = out[key] + item
    
    for key in out.keys():
        out[key]=list(set(out[key]))
    return out


def save_checkpoint(model, epoch, checkpoint_dir, buffer, max_to_keep=10):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(checkpoint_dir, 'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)
    buffer.append(filename)
    if len(buffer)>max_to_keep:
        os.remove(buffer[0])
        del(buffer[0])

    return buffer


def restore_checkpoint(model, checkpoint_dir, device, force=False, pretrain=False):
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
        # inp_epoch = int(input())
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


def restore_best_checkpoint(epoch, model, checkpoint_dir, device):
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


def clear_checkpoint(checkpoint_dir):
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def evaluation(args, data, model, epoch, base_path, evaluator, name="valid"):
    # Evaluate with given evaluator

    ret, _ = evaluator.evaluate(model)

    n_ret = {"recall": ret[1], "hit_ratio": ret[5], "precision": ret[0], "ndcg": ret[3], "mrr":ret[4], "map":ret[2]}

    perf_str = name+':{}'.format(n_ret)
    print(perf_str)
    with open(base_path + 'stats.txt', 'a') as f:
        f.write(perf_str + "\n")
    # Check if need to early stop (on validation)
    is_best=False
    early_stop=False
    if name=="test":
        if ret[1] > data.best_valid_recall:
            data.best_valid_epoch = epoch
            data.best_valid_recall = ret[1]
            data.patience = 0
            is_best=True
        else:
            data.patience += 1
            if data.patience >= args.patience:
                print_str = "The best performance epoch is % d " % data.best_valid_epoch
                print(print_str)
                early_stop=True

    return is_best, early_stop, n_ret

def ensureDir(dir_path):

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def checktensor(tensor):
    t=tensor.detach().cpu().numpy()
    if np.max(np.isnan(t)):        
        idx=np.argmax(np.isnan(t))
        return idx
    else:
        return -1

def fix_seeds(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def visualize_and_save_log(file_dir, dataset_name, show=False):
    if(dataset_name == "tencent_synthetic"):
        pass
    else:
        valid_recall, valid_ndcg, test_recall, test_ndcg = [], [], [], []

        with open(file_dir, 'r') as f:
            # count = 0
            for line in f:
                line = line.split(' ')
                if("valid" in line[0]):
                    valid_recall.append(float(line[1][:-1]))
                    valid_ndcg.append(float(line[7][:-1]))
                if("test" in line[0]):
                    test_recall.append(float(line[1][:-1]))
                    test_ndcg.append(float(line[7][:-1]))

        epochs = list(range(0, len(valid_recall)))
        epochs = [i*5 for i in epochs]
        result = pd.DataFrame({'epochs': epochs, 'valid_recall': valid_recall, 'test_recall': test_recall, 'valid_ndcg': valid_ndcg, 'test_ndcg': test_ndcg})
        df = result.iloc[:-1, :]

        fig=plt.figure()
        x = df.epochs
        y1 = df.valid_recall
        y2 = df.test_recall
        print(max(y1), max(y2), 1.1*max(y1), 1.1*max(y2))

        ax1=fig.subplots()
        ax2=ax1.twinx()    
        ax1.plot(x,y1,'g-', label='valid_recall')
        ax2.plot(x,y2,'b--', label='test_recall')
        ax1.set_ylim(min(y1), 1.15*(max(y1)-min(y1))+min(y1))
        ax2.set_ylim(min(y2), 1.15*(max(y2)-min(y2))+min(y2))

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('valid_recall')
        ax2.set_ylabel('test_recall')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        base_path = file_dir[:-9]
        save_path = base_path + "/train_log.png"
        plt.savefig(save_path)
        if(show):
            plt.show()
        save_path = base_path + "/train_log.csv"
        result.to_csv(save_path, index=False)
        
def csr_to_torch_sparse(csr_mat):
    csr_mat = csr_mat.tocoo() 
    row = torch.from_numpy(csr_mat.row).long()
    col = torch.from_numpy(csr_mat.col).long()
    indices = torch.stack([row, col], dim=0)
    values = torch.from_numpy(csr_mat.data).float()
    shape = torch.Size(csr_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)