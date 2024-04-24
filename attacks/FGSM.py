"""
Perform a FGSM attack + evaluation
- on single file
- on batches

author: wwang
"""


from src.utils import *
from attacks_utils import load_spec_model, FGSM_attack



if __name__ == '__main__':

    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)

    # load the model + load state dict + set to eval
    model = load_spec_model(device=device, config=config)
    print('Model loaded')
    model.eval()

    # create the attack object given an epsilon
    epsilon = 0.3
    FGSM_attack = FGSM_attack(config, model, device, epsilon)
    print(f'Will perform an untargeted FGSM attack with epsilon = {epsilon}')

    # perform the FGSM on a single file + save spec + save audio
    file_index = 0
    FGSM_attack.attack_single_cached_spec(index=file_index)

    # TODO succesfully saved the spec in the correct folder
    # TODO now we have to save the perturbed audio