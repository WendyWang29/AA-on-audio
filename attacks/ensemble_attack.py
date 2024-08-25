import matplotlib.pyplot as plt
import torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from src.utils import *
import os
import gc
import librosa
from src.resnet_model import SpectrogramModel
from src.SENet.SENet_model import se_resnet34_custom
from src.resnet_utils import LoadAttackData_ResNet
from check_attacks_utils import get_model_prediction, compute_confidence, get_GT_label


def plot_3d_grad(grad, model):
    x = np.arange(grad.shape[1])
    y = np.arange(grad.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, grad, cmap='viridis')

    ax.set_xlabel('Y axis')
    ax.set_ylabel('X axis')
    ax.set_zlabel('Z axis')
    plt.title(f'{model} gradient')
    plt.show()

def Ensemble_Attack_SingleFile(file_number, epsilon, eval_path, config_SENet, config_ResNet, df_eval, device):
    GT_label = get_GT_label(file_number, eval_path)

    # load the original audio file
    clean_path = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_eval/flac'
    file_number = str(file_number)
    clean_path = os.path.join(clean_path, f'LA_E_{file_number}' + '.flac')
    audio, _ = librosa.load(clean_path, sr=None, duration=240, mono=True)

    # clean spec to batch and predictions
    resnet_pred, resnet_pred_label, clean_spec = get_model_prediction(eval_model=ResNet_model,
                                                                      pert_audio=audio,
                                                                      device=device,
                                                                      flag=0)

    senet_pred, senet_pred_label, clean_spec = get_model_prediction(eval_model=SENet_model,
                                                                    pert_audio=audio,
                                                                    device=device,
                                                                    flag=1)
    print(f'---> File number: {file_number}\n',
          f'--> GT label: {GT_label}\n',
          f'--> Predicted label ResNet: {resnet_pred_label} --- Confidence: {compute_confidence(resnet_pred):.2f} %\n',
          f'--> Predicted label SENet: {senet_pred_label} --- Confidence SENet: {compute_confidence(senet_pred):.2f} %')

    # creating the (mini) batches
    batch_x = torch.from_numpy(clean_spec).unsqueeze(dim=0).to(device)
    batch_z = batch_x.clone().to(device)
    batch_y = torch.tensor([int(GT_label)]).to(device)

    L = nn.NLLLoss()

    # ResNet
    batch_x.requires_grad = True
    out_res = ResNet_model(batch_x)
    loss_res = L(out_res, batch_y)
    ResNet_model.zero_grad()
    loss_res.backward()
    grad_res = batch_x.grad.data

    # SENet
    batch_z.requires_grad = True
    out_sen = SENet_model(batch_z.unsqueeze(dim=1))
    loss_sen = L(out_sen, batch_y)
    SENet_model.zero_grad()
    loss_sen.backward()
    grad_sen = batch_z.grad.data

    grad_res_c = grad_res.clone().squeeze().detach().cpu().numpy()
    plot_3d_grad(grad=grad_res_c, model='ResNet')
    grad_sen_c = grad_sen.clone().squeeze().detach().cpu().numpy()
    plot_3d_grad(grad=grad_sen_c, model='SENet')

    grad_avg = (grad_sen + grad_res) / 2
    perturbed_batch = batch_x + epsilon * grad_avg.sign()

    out_avg_res = ResNet_model(perturbed_batch)
    print(out_avg_res)







    print('uwu')






if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_ResNet_path = '../config/residualnet_train_config.yaml'
    config_SENet_path = '../config/SENet.yaml'

    config_ResNet = read_yaml(config_ResNet_path)
    config_SENet = read_yaml(config_SENet_path)

    df_eval = pd.read_csv(os.path.join('..', config_ResNet['df_eval_path']))

    # load the models
    ResNet_model = SpectrogramModel().to(device)
    ResNet_model.load_state_dict(torch.load(os.path.join('..', config_ResNet['model_path_spec']), map_location=device), strict=False)
    ResNet_model.eval()
    SENet_model = se_resnet34_custom(num_classes=2).to(device)
    SENet_model.load_state_dict(torch.load(os.path.join('..', config_SENet['model_path_spec']), map_location=device), strict=False)
    SENet_model.eval()
    print('Models loaded...\n')

    eval_path = os.path.join('..', config_SENet['df_eval_path'])  # eval dataset of ASVSpoof2019

    epsilon = 1.0
    file_number = 2520601

    Ensemble_Attack_SingleFile(file_number, epsilon, eval_path, ResNet_model, SENet_model, df_eval, device)
