"""
FGSM adversarial attack on whole datasets
"""
from src.utils import *
from attacks_utils import load_spec_model, FGSMAttack




if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = '../config/residualnet_train_config.yaml'
    config = read_yaml(config_path)

    # set the dataset to perturb = create list of path to files
    df_eval = pd.read_csv(os.path.join('..', config["df_eval_path"]))

    # load the model on device  + load state dict + set to eval
    model = load_spec_model(device=device, config=config)
    print('Model loaded')
    model.eval()

    # set the perturbation
    epsilon = 0.8

    FGSM_attack = FGSMAttack(epsilon, config, model, device)
    FGSM_attack.attack_dataset(eval_csv=df_eval)

