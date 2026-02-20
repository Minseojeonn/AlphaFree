from models.AlphaFreeRecDemo import AlphaFreeRecDemo
from models.base.parse import parse_args
import gdown
import torch
import os


if __name__ == '__main__':
    
    #download demo weights if not exist
    weight_file_path = "./weights/inference_demo/weights.pth.tar"
    if not os.path.exists(weight_file_path):
        os.makedirs("./weights/inference_demo", exist_ok=True)
        file_id = "1dAbq-vfolKsQMckrS86czIJIBJMXBmHg"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, weight_file_path, quiet=False)
    
    #demo model inference load
    args, special_args = parse_args() 
    model = AlphaFreeRecDemo(args)
    checkpoint = torch.load(weight_file_path)
    state_dict = checkpoint['state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("mlp_origin.")}
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.cuda(args.cuda)
    
    #inference demo
    model.eval()
    with torch.no_grad():
        while True:
            user_input = input("Enter the set of interaction items. (comma-separated, e.g., 40, 30) or type ‘exit’ to quit:")
            if user_input.lower() == 'exit':
                break
            try:
                user_input = [int(uid.strip()) for uid in user_input.split(',')]
                with torch.no_grad():
                    predictions = model.predict(user_input)
                    print(f"Top-20 item predictions for interactions without masking {user_input} : {predictions}")
            except ValueError:
                print("Invalid input. Please enter a list of integers separated by commas.")
        

    
    
