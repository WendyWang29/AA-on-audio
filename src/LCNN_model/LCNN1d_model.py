import torch
import torch.nn as nn
import torch.nn.functional as F # for the log softmax

'''
LCNN_model = lightweight CNN
with MFM (max feature maps) which work similarly to ReLU
see https://arxiv.org/pdf/1511.02683
'''

class PowerSpectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, win_length=2048, eps=1e-20):
        super(PowerSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.eps = eps
    def forward(self, x):
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=False)
        mag_spec = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2 + self.eps)
        pow_spec = mag_spec ** 2
        db_spec = 10 * torch.log10(pow_spec + self.eps)
        return db_spec

class LCNN1D(nn.Module):
    def __init__(self, num_class=2):
        super(LCNN1D, self).__init__()
        self.powerspec = PowerSpectrogram()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                               padding=(2, 2), stride=(1, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                               padding=(0, 0), stride=(1, 1))
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                               padding=(1, 1), stride=(1, 1))
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)

        self.conv11 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv17 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                padding=(0, 0), stride=(1, 1))
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))
        # self.fc29 = nn.Linear(32*20*3, 160) #5#nn.Linear(12*32*4*16,2*64)
        self.fc29 = nn.Linear(32 * 16 * 8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.7)

        self.fc32 = nn.Linear(64, num_class)

    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)

    def mfm3(self, x):
        n, c, y, z = x.shape
        out1, out2, out3 = torch.chunk(x, 3, 1)
        res1 = torch.max(torch.max(out1, out2), out3)
        tmp1 = out1.flatten()
        tmp1 = tmp1.reshape(len(tmp1), -1)
        tmp2 = out2.flatten()
        tmp2 = tmp2.reshape(len(tmp2), -1)
        tmp3 = out3.flatten()
        tmp3 = tmp3.reshape(len(tmp3), -1)
        res2 = torch.cat((tmp1, tmp2, tmp3), 1)
        res2 = torch.median(res2, 1)[0]
        res2 = res2.reshape(n, -1, y, z)
        return torch.cat((res1, res2), 1)

    def forward(self, x):
        x = self.powerspec(x)
        x = self.conv1(x.unsqueeze(1))
        x = self.mfm2(x)
        # ------------------------
        x = self.maxpool3(x)
        # ------------------------
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        # ------------------------
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        # ------------------------
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        # ------------------------
        x = self.maxpool16(x)
        # ------------------------
        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        # ------------------------
        x = self.maxpool28(x)

        x = x.view(-1, 32 * 16 * 8)
        x = self.mfm2((self.fc29(x)))
        x = self.batchnorm31(x)
        logits = self.fc32(x)

        return F.log_softmax(logits, dim=1)