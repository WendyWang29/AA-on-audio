from src.resnet_model import SpectrogramModel
from src.resnet_utils import get_features
from src.utils import *
import numpy as np
import matplotlib.pyplot as plt

def load_spec_model(device, config):
    """
    Load the spectrogram model - pre-trained
    :param device: GPU or CPU
    :param config: config file path
    :return: model
    """
    resnet_spec_model = SpectrogramModel().to(device)
    resnet_spec_model.load_state_dict(torch.load(config['model_path_spec'], map_location=device))
    return resnet_spec_model

def plot_spec(spec, index):
    plt.figure(figsize=(10, 6))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel('Time frame')
    plt.ylabel('Frequency bin')
    plt.title(f'Magnitude spectrogram for file {index} ')
    plt.show()

def eval_one_file_spec(index, config):
    """
    Evaluate the spectrogram model on one single file taken from eval list
    :param index: the index of the single file you want to evaluate
    :param config: config file
    :return: predictions as a tensor
    """

    # read the .csv and get the list of paths to the files
    df_eval = pd.read_csv(config['df_eval_path'])
    file_eval = list(df_eval['path'])

    # get the single wav path
    file = file_eval[index]

    # get the feature (the cached spec is a ndarray: (1025,41) )
    X = get_features(wav_path=file,
                     features=config['features'],
                     args=config,
                     X=None,
                     cached=True,
                     force=False)

    # plot the spectrogram
    plot_spec(X, index)
    #TODO check if time is correct

    # transform into a mini batch and to a tensor
    X_batch = np.expand_dims(X, axis=0)  # shape ndarray (1,1025,41)
    X_batch_tensor = torch.from_numpy(X_batch).to(device) # tensor (1,1025,41)

    # get the prediction
    pred = model(X_batch_tensor)

    return pred


if __name__ == '__main__':
    seed_everything(1234)
    set_gpu(-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get the config settings
    config_path = 'config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    # load the spectrogram model
    model = load_spec_model(device, config_res)
    model.eval()

    # evaluate one single file
    pred = eval_one_file_spec(index=0, config=config_res)

    print(pred)
