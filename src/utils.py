import os
import torch
import random
import GPUtil
# import tensorflow as tf
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import torch.nn as nn


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)

    :param id: if specified, corresponds to the GPU index desired.
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # # Set memory growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    return device


def prepare_asvspoof_data(config):

    data_dir_2019 = '/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols'
    data_eval_2021 = '/nas/public/dataset/asvspoof2021/DF_cm_eval_labels.txt'
    files = [os.path.join(data_dir_2019, 'ASVspoof2019.LA.cm.train.trn.txt'),
        os.path.join(data_dir_2019, 'ASVspoof2019.LA.cm.dev.trl.txt'), data_eval_2021]

    audio_dir_2019 = '/nas/public/dataset/asvspoof2019/LA'
    audio_dir_2021 = '/nas/public/dataset/asvspoof2021/ASVspoof2021_DF_eval/flac/'
    set_dirs = [os.path.join(audio_dir_2019, 'ASVspoof2019_LA_train/flac/'),
                os.path.join(audio_dir_2019, 'ASVspoof2019_LA_dev/flac/'), audio_dir_2021]

    save_paths = [config['df_train_path'], config['df_dev_path'], config['df_eval_path']]

    for file_path, set_dir, save_path in zip(files, set_dirs, save_paths):

        txt_file = pd.read_csv(file_path, sep=' ', header=None)
        txt_file = txt_file.replace({'bonafide': 0, 'spoof': 1})

        txt_file.iloc[:,1] = set_dir + txt_file.iloc[:,1].astype(str) + '.flac'

        if not file_path == data_eval_2021:
            df = txt_file[[1, 4]]
            df = df.rename({1: 'path', 4: 'label'}, axis='columns')
        else:
            df = txt_file[[1, 5]]
            df = df.rename({1: 'path', 5: 'label'}, axis='columns')

        df.to_csv(save_path)


def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def sigmoid(x, factor=1):
    """
    Compute sigmoid function.

    :param x: input signal
    :param factor: sigmoid parameter
    :return: sigmoid(x)
    :rtype np.array
    """
    z = 1 / (1 + np.exp(-factor*x))
    return z


def plot_roc_curve(labels, pred, legend=None):
    """
    Plot ROC curve.

    :param labels: groundtruth labels
    :type labels: list
    :param pred: predicted score
    :type pred: list
    :param legend: if True, add legend to the plot
    :type legend: bool
    :return:
    """
    # labels and pred bust be given in (N, ) shape

    def tpr5(y_true, y_pred):
        fpr, tpr, thr = roc_curve(y_true, y_pred)
        fp_sort = sorted(fpr)
        tp_sort = sorted(tpr)
        tpr_ind = [i for (i, val) in enumerate(fp_sort) if val >= 0.1][0]
        tpr01 = tp_sort[tpr_ind]
        return tpr01

    lw = 3

    fpr, tpr, _ = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    print('TPR5 = {:.3f}'.format(tpr5(labels, pred)))
    print('AUC = {:.3f}'.format(rocauc))
    print('EER = {:.3f}'.format(eer))
    print()
    if legend:
        plt.plot(fpr, tpr, lw=lw, label='$\mathrm{' + legend + ' - AUC = %0.2f}$' % rocauc)
    else:
        plt.plot(fpr, tpr, lw=lw, label='$\mathrm{AUC = %0.2f}$' % rocauc)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel(r'$\mathrm{False\;Positive\;Rate}$', fontsize=18)
    plt.ylabel(r'$\mathrm{True\;Positive\;Rate}$', fontsize=18)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, normalize=False, cmap=plt.cm.Blues):
    """
    Plot confusion matrix.

    :param y_true: ground-truth labels
    :type y_true: list
    :param y_pred: predicted labels
    :type y_pred: list
    :param normalize: if set to True, normalise the confusion matrix.
    :type normalize: bool
    :param cmap: matplotlib cmap to be used for plot
    :type cmap:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    classes = ['$\it{Real}$','$\it{Fake}$']
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    fsize = 25  # fontsize
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, clim=(0,1))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fsize)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           )
    ax.set_xlabel('$\mathrm{True\;label}$', fontsize=fsize)
    ax.set_ylabel('$\mathrm{Predicted\;label}$', fontsize=fsize)
    ax.set_xticklabels(classes, fontsize=fsize)
    ax.set_yticklabels(classes, fontsize=fsize)
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format('$\mathrm{' + str(format(cm[i, j], fmt)) + '}$'),
                    ha="center", va="center",
                    fontsize=fsize,
                    color="white" if np.array(cm[i, j]) > thresh else "black")
    fig.tight_layout()
    # plt.show()

    return ax


def reconstruct_from_pred(pred_array, win_len, hop_size, fs=16000):
    """
    Create a score array with length equal to the original signal length starting from predictions aggregated on
    rectangular windows.

    :param pred_array: aggregated prediction array
    :type pred_array: list
    :param win_len: length of the window used for aggregation
    :type win_len: int
    :param hop_size: length of the hop used for aggregation
    :type hop_size: int
    :param fs: sampling frequency
    :type fs: int
    :return: reconstructed array
    """

    pred_array = np.array(pred_array)
    audio_shape = (len(pred_array)-1) * hop_size * fs + win_len * fs

    window_pred = np.zeros((len(pred_array), int(audio_shape)))
    for idx, pred in enumerate(pred_array):
        window_pred[idx, int(idx*hop_size*fs):int((idx*hop_size+win_len)*fs)] = pred

    window_pred = np.nanmean(np.where(window_pred != 0, window_pred, np.nan), 0)

    return window_pred


def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
