from models.base.RS import AlphaFree_RS
from parse import parse_args
from utils import fix_seeds

if __name__ == '__main__':
    args, special_args = parse_args()
    print(args)
    fix_seeds(args.seed) 
    if args.model_name == 'AlphaFree':
        RS = AlphaFree_RS(args, special_args)
        RS.execute()
    else:
        print("Model not implemented!")
    
    
