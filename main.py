from models.base.RS import AlphaFreeRS
from models.base.parse import parse_args
from models.base.utils import fix_seeds
from models.base.data import AlphaFreeData 

if __name__ == '__main__':
    args, special_args = parse_args()
    print(args)
    fix_seeds(args.seed)
    
    # Phase preprocessing : data preprocessing (Language Representation Generation, Augmentations ..)
    if args.phase == 'preprocessing':
        data = AlphaFreeData(args)
    
    # Phase inference : inference using pre-trained model 
    elif args.phase == 'inference':
        RS = AlphaFreeRS(args, special_args)
        RS.inference_only()
        
    # Phase train : training model from scratch
    elif args.phase == 'train':
        RS = AlphaFreeRS(args, special_args)
        RS.execute()
        
    # Something wrong, please check phase
    else:
        print("Something wrong with phase, we only support \"train\", \"preprocessing\", \"inference\"")
    
    
