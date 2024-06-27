"""
Adversarial attacks using RawNet2
options:
- attack single file
- attack dataset #TODO
"""
from src.utils import *
from attack_classes import RawNetAttack
import logging



if __name__ == '__main__':
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    mode: 'single' or 'dataset'
    epsilon: amount of perturbation
    index: index of the file referring to df_eval_19.csv
    """

    mode = 'single'
    epsilon = 0.001
    index = 0

    attack = RawNetAttack(device=device, mode=mode)
    attack.BIMc_RawNet(epsilon=epsilon, index=index)


    # config_path = '../config/rawnet2.yaml'
    # config = read_yaml(config_path)
    #
    # # set the dataset to perturb = create list of path to files
    # df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))
    #
    # # load the model on device  + load state dict + set to eval
    # rawnet_model = RawNet(config['model'], device)
    # rawnet_model = rawnet_model.to(device)
    # rawnet_model.load_state_dict(torch.load(os.path.join('..', config['model_path_spec']), map_location=device))
    # rawnet_model.eval()
    #
    # # set the perturbation
    # epsilon = 0.2
    #
    # FGSM_attack = FGSMAttack(epsilon, config, rawnet_model, device)
    # #FGSM_attack.attack_dataset_RawNet(eval_csv=df_eval)
    # FGSMAttack.attack_single()