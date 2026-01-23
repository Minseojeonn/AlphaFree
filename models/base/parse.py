import argparse
import os
import yaml


def build_parser():
    parser = argparse.ArgumentParser()

    # Phase Args
    parser.add_argument('--phase', type=str, default='train',
                        choices=['preprocessing', 'train', 'inference'],
                        help='The phase to run the model in.')
    parser.add_argument('--dataset', type=str, default='amazon_movie',
                        help='Dataset name.')
    # General Args
    parser.add_argument('--model_name', type=str, default='AlphaFree',
                        help='model name.')    
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--test_only', action="store_true",
                        help='Whether to test only.')
    parser.add_argument('--clear_checkpoints', action="store_true", default=True,
                        help='Whether clear the earlier checkpoints.')
    parser.add_argument('--saveID', type=str, default='Saved',
                        help='Specify model save path. Description of the experiment')
    parser.add_argument('--seed', type=int, default=101,
                        help='Random seed.')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--verbose', type=float, default=5,
                        help='Interval of evaluation.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping point.')

    # Model Args
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay for optimizer.')

    
    parser.add_argument('--Ks', type = int, default= 20,
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--neg_sample',type=int,default=256)
    parser.add_argument('--infonce', type=int, default=1,
                help='whether to use infonce loss or not')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--max2keep', type=int, default=1,
                        help='max checkpoints to keep')
    
    parser.add_argument('--tau_r', type=float, default=0.1,
                    help='tau_r')
    parser.add_argument('--lm_model', type=str, default='llama',
                choices=['v3', 'llama'],
                help='The base language model')
    parser.add_argument('--tau_a', type=float, default=0,
                help='tau_a')
    parser.add_argument('--K_c', type=int, default=0,
                help='K_c')
    parser.add_argument('--lambda_align', type=float, default=0.07,
                        help='align cl loss temperature')

    return parser

def load_dataset_config(config_dir: str, dataset: str):
    path = os.path.join(config_dir, f"{dataset}.yaml")
    if not os.path.exists(path):
        return {}, path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict at top-level: {path}")
    return cfg, path


def parse_args():
    # 1) minimal parse to know dataset/config_dir
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument('--dataset', type=str, default='amazon_movie')
    mini.add_argument('--config_dir', type=str, default='./configs')
    mini_args, _ = mini.parse_known_args()

    # 2) load config by dataset
    cfg, cfg_path = load_dataset_config(mini_args.config_dir, mini_args.dataset)

    # 3) build full parser and set defaults from config
    parser = build_parser()
    parser.add_argument('--config_dir', type=str, default=mini_args.config_dir)
    parser.add_argument('--config_path', type=str, default=cfg_path, help='Loaded config path (auto).')

    # config에 있는 key만 parser에 존재하는 옵션이면 반영
    valid_keys = {a.dest for a in parser._actions}
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    parser.set_defaults(**filtered)
    parser.set_defaults(dataset=mini_args.dataset)

    # 4) final parse (CLI가 config 덮어씀)
    args = parser.parse_args()

    # special args(옵션 목록) 필요하면 그대로 유지 가능
    special_args = sorted(list(filtered.keys()))
    return args, special_args