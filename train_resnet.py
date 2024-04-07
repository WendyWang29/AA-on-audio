import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

logger = logging.getLogger("add_challenge")
logger.setLevel(logging.INFO)
import pandas as pd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from src.resnet_model import SpectrogramModel, MFCCModel
from src.resnet_utils import LoadTrainData_ResNet, train_epoch_resnet, evaluate_accuracy_resnet, get_loss_resnet, evaluate_metrics
from src.utils import *
from sklearn import model_selection


def main(config):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert config['features'] in ['spec', 'mfcc'], 'Not supported feature'
    model_tag = 'model_{}_{}_{}_{}'.format(config['features'], config['num_epochs'], config['batch_size'], config['lr'])
    model_save_path = os.path.join(config['model_folder'], model_tag)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if config['features'] == 'mfcc':
        model_cls = MFCCModel
    elif config['features'] == 'spec':
        model_cls = SpectrogramModel
    else:
        return

    model = model_cls().to(device)

    df_train = pd.read_csv(config["df_train_path"])
    df_dev = pd.read_csv(config["df_dev_path"])

    d_label_trn = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])
    train_set = LoadTrainData_ResNet(list_IDs=file_train, labels=d_label_trn, win_len=config['win_len'], config=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=15)
    del train_set, d_label_trn

    # define validation dataloader
    d_label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])
    dev_set = LoadTrainData_ResNet(list_IDs=file_dev, labels=d_label_dev, win_len=config['win_len'], config=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=True, num_workers=15)
    del dev_set, d_label_dev

    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 0
    best_loss = 1
    early_stopping = 0
    for epoch in range(config['num_epochs']):
        if early_stopping < config['early_stopping']:
            running_loss, train_accuracy = train_epoch_resnet(train_loader, model, config['lr'], device)
            valid_accuracy = evaluate_accuracy_resnet(dev_loader, model, device)
            valid_auc, valid_eer, valid_accuracy = evaluate_metrics(dev_loader, model, device)
            valid_loss = get_loss_resnet(dev_loader, model, device)
            writer.add_scalar('train_accuracy', train_accuracy, epoch)
            writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
            writer.add_scalar('train_loss', running_loss, epoch)
            writer.add_scalar('valid_loss', valid_loss, epoch)
            logger.info(f"Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: {valid_loss:.5f} - Train Acc: {train_accuracy:.2f} - Val Acc: {valid_accuracy:.2f}")

            if valid_loss < best_loss:
                logger.info(f"Best model found at epoch {epoch}")
                torch.save(model.state_dict(), os.path.join(model_save_path, config['save_trained_name']))
                early_stopping = 0
            else:
                early_stopping += 1
            best_loss = min(valid_loss, best_loss)
        else:
            logger.info(f"Training stopped after {epoch} epochs - Best Val Acc {best_acc:.2f}")
            break

if __name__ == '__main__':

    seed_everything(1234)
    set_gpu(-1)

    config_path = 'config/residualnet_train_config.yaml'
    config_res = read_yaml(config_path)

    main(config_res)
