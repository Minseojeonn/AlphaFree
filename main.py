from models.base.RS import AlphaFreeRS
from AlphaFree.models.base.parse import parse_args
from models.base.utils import fix_seeds

if __name__ == '__main__':
    args, special_args = parse_args()
    print(args)
    fix_seeds(args.seed) 
    if args.model_name == 'AlphaFree':
        RS = AlphaFreeRS(args, special_args)
        RS.execute()
    else:
        print("Model not implemented!")
    
    
