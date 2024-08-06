"""
Adversarial attacks using ResNet
options:
- attack single file
- attack dataset #TODO
"""

from src.utils import *
from attack_classes import ResNetAttack



if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """
    mode: 'single' or 'dataset'
    epsilon: amount of perturbation
    index: index of the file referring to df_eval_19.csv
    """

    mode = 'single'
    epsilon = 1.0
    index = 1

    attack = ResNetAttack(device=device, mode=mode)

    '''
    FGSM Cut Attack
    a variation of the FGSM attack on ResNet on which the gradient gets cut and tiled
    like the spectrogram which is used as input for the model
    '''
    #attack.BIMCut_ResNet(epsilon, index)




    '''
    FGSM attack
    normal FGSM attack which I used for generating all the perturbed dataset
    '''
    attack.FGSM_ResNet_c(epsilon=epsilon, index=index)

    '''
    SSA attack (spectrum simulation attack)
    '''
    #attack.SSA_IFGSM_ResNet(epsilon=epsilon, index=index)


