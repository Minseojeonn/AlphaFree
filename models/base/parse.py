import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Phase Args
    parser.add_argument('--phase', type=str, default='train',
                        choices=['preprocessing', 'train', 'inference'],
                        help='The phase to run the model in.')

    # General Args
    parser.add_argument('--model_name', type=str, default='AlphaFree',
                        help='model name.')
    parser.add_argument('--dataset', nargs='?', default='amazon_movie',
                        help='yc, ks, rr')
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
    args, _ = parser.parse_known_args()

    
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
    args, _ = parser.parse_known_args()
    
    parser.add_argument('--tau_r', type=float, default=0.1,
                    help='tau_r')
    parser.add_argument('--lm_model', type=str, default='v3',
                choices=['v3', 'llama'],
                help='The base language model')
    parser.add_argument('--tau_a', type=float, default=0,
                help='tau_a')
    parser.add_argument('--K_c', type=int, default=0,
                help='K_c')
    parser.add_argument('--lambda_align', type=float, default=0.07,
                        help='align cl loss temperature')

    args_full, _ = parser.parse_known_args()
    special_args = list(set(vars(args_full).keys()) - set(vars(args).keys()))
    special_args.sort()
    return args_full, special_args
