"""
Perform a FGSM attack + evaluation
- on single file
- on batches

author: wwang
"""


from src.utils import *
from attacks_utils import load_spec_model, FGSMAttack



if __name__ == '__main__':
    '''
    ########### PRELIMS ###########
    '''

    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)

    # load the model + load state dict + set to eval
    model = load_spec_model(device=device, config=config)
    print('Model loaded')
    model.eval()

    '''
    ########### ATTACK ONE SINGLE FILE ###########
    '''

    file_index = 0
    epsilon = 1.5

    # create the attack (on single file) object given an epsilon
    FGSM_attack = FGSMAttack(epsilon, config, model, device)

    # perform the attack + save perturbed audio and spec
    """
    VALID ATTACKS:
    'FGSM': normal attack
    """
    FGSM_attack.attack_single(file_index, 'FGSM')
    # TODO find black box attack etc etc