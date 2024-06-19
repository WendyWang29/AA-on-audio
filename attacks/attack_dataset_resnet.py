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

    mode = 'dataset'
    epsilon = 3.0
    index = None

    '''
    SSA attack
    '''
    attack = ResNetAttack(device=device, mode=mode)
    attack.SSA_IFGSM_ResNet(epsilon=epsilon, index=index)
