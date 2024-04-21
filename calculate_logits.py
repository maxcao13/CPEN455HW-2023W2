from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

# Based off of classification_evaluation.py to compute logits for the final hugging face submission
def get_label(model, model_input, device):
    batch_size = model_input.shape[0]
    
    losses = torch.zeros(batch_size, NUM_CLASSES, device=device)

    for label in range(NUM_CLASSES):
        labels = torch.full((batch_size,), label, dtype=torch.long, device=device)

        logits = model(model_input, labels)

        log_prob = discretized_mix_logistic_loss(model_input, logits, sum_all=False)

        losses[:, label] = log_prob

    return losses.detach().cpu().numpy()

def classifier(model, data_loader, device):
    model.eval()
    logits = np.empty((0, 4))
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        logit = get_label(model, model_input, device) # should be [B, num_classes]
        logits = np.concatenate((logits, logit))
    
    return logits
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=1, nr_filters=80, input_channels=3, nr_logistic_mix=10)
    
    model = model.to(device)
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth'))
    model.eval()
    print('model parameters loaded')
    logits = classifier(model = model, data_loader = dataloader, device = device)
    logits.dump('test_logits.npy')
    print(f"Sucessfully dumped to test_logits!")
        
        