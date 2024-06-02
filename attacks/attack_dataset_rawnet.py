"""
FGSM adversarial attack on whole datasets
using RawNet2
in attack_utils.py --> attack_dataset method --> change the flag model='RawNet'
"""
from src.utils import *
import os
from src.rawnet2_model import RawNet
from attacks_utils import FGSMAttack




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = '../config/rawnet2.yaml'
    config = read_yaml(config_path)

    # set the dataset to perturb = create list of path to files
    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))

    # load the model on device  + load state dict + set to eval
    rawnet_model = RawNet(config['model'], device)
    rawnet_model = rawnet_model.to(device)
    rawnet_model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device))
    rawnet_model.eval()

    # set the perturbation
    epsilon = 0.005

    FGSM_attack = FGSMAttack(epsilon, config, rawnet_model, device)
    FGSM_attack.attack_dataset_RawNet(eval_csv=df_eval)