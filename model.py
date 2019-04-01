import torch.nn as nn
import torch.nn.functional as F
import torch
import utils_PyTorch as utils

class Text_CNN_list(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, mode, word_dim, vocab_size, out_dim, filters, filter_num, dropout_prob, wv_matrix, in_channel=1, mid=1024, one_layer=False):
        super(Text_CNN_list, self).__init__()
        self.mode = mode
        self.word_dim = word_dim
        self.vocab_size = vocab_size
        self.out_dim = out_dim
        self.filters = filters
        self.filter_num = filter_num
        self.dropout_prob = dropout_prob
        self.in_channel = in_channel
        self.one_layer = one_layer

        assert (len(self.filters) == len(self.filter_num))
        self.embedding = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
        if self.mode == "static" or self.mode == 'non-static' or self.mode == 'multichannel':
            self.wv_matrix = wv_matrix
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv_matrix))
            if self.mode == 'static':
                self.embedding.weight.requires_grad = False
            elif self.mode == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.word_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.wv_matrix))
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2

        self.convs1 = nn.ModuleList([nn.Conv2d(self.in_channel, out_channel, (K, self.word_dim)) for out_channel, K in zip(self.filter_num, self.filters)])
        # self.dropout = nn.Dropout(self.dropout_prob)
        # self.fc = nn.Linear(sum(self.filter_num), self.out_dim)
        if not one_layer:
            self.fc1 = nn.Linear(sum(self.filter_num), mid)
            self.fc2 = nn.Linear(mid, out_dim)
        # self.fc2 = nn.Linear(1024, out_dim)

    def forward(self, x):
        out = self.embedding(x).unsqueeze(1)

        if self.mode == 'multichannel':
            out2 = self.embedding2(x).unsqueeze(1)
            out = torch.cat((out, out2), 1)

        out = [F.relu(_conv(out)).squeeze(3) for _conv in self.convs1]
        out = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in out]
        out1 = torch.cat(out, 1)
        # out = self.dropout(out)
        if not self.one_layer:
            out2 = F.relu(self.fc1(out1))
            # out3 = self.fc2(out1)
            # return [out1, out3]

            out3 = self.fc2(out2)
            return [out1, out2, out3]
        else:
            return [out1]

class Dense_Net(nn.Module):
    """Generator for transfering from svhn to mnist"""

    def __init__(self, input_dim=28*28, out_dim=20, mid=1024, one_layer=False):
        super(Dense_Net, self).__init__()
        self.one_layer = one_layer
        self.fc1 = nn.Linear(input_dim, mid)
        # self.fc2 = nn.Linear(mid_num, out_dim)
        if not one_layer:
            self.fc2 = nn.Linear(mid, mid)
            self.fc3 = nn.Linear(mid, out_dim)

        # self.fc1 = nn.Linear(input_dim, 4096)
        # # self.fc2 = nn.Linear(mid_num, out_dim)
        # if not one_layer:
        #     self.fc2 = nn.Linear(4096, 4096)
        #     self.fc3 = nn.Linear(4096, out_dim)

        # decoding blocks
        # self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        # self.deconv2 = deconv(conv_dim, 1, 4, bn=False)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        if not self.one_layer:
            out2 = F.relu(self.fc2(out1))
            out3 = self.fc3(out2)
            return [out1, out2, out3]
        else:
            return [out1]


        # out1 = F.relu(self.fc1(x))
        # out2 = F.relu(self.fc2(out1))
        # out3 = self.fc3(out1)
        # return [out1, out3]
        # # out2 = self.fc3(out1)
        # # return [out1, out2]Image_CNN_Net

class D(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, dim=20, view=2):
        super(D, self).__init__()
        self.dim = dim
        self.n_view = view
        mid_dim = 128
        self.fc1 = nn.Linear(dim, mid_dim)
        # self.fc2 = linear(128, 64)
        n_out = self.n_view
        self.fc2 = nn.Linear(mid_dim, n_out)

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.softmax(out, dim=1)


class Half_MNIST_CNN_Net(nn.Module):
    def __init__(self, output_shape, in_channel=1, mid=1024, one_layer=False):
        super(Half_MNIST_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 5 * 1, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not self.one_layer:
            self.fc2 = nn.Linear(mid, output_shape)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out4 = out4.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out5 = F.relu(self.fc1(out4))
        if not self.one_layer:
            out6 = self.fc2(out5)
            return [out1, out2, out3, out4, out5, out6]
        else:
            return [out5]

class MNIST_CNN_Net(nn.Module):
    def __init__(self, output_shape, in_channel=1, mid=1024, one_layer=False):
        super(MNIST_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 4 * 4, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not self.one_layer:
            self.fc2 = nn.Linear(mid, output_shape)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out4 = out4.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out5 = F.relu(self.fc1(out4))
        if not self.one_layer:
            out6 = self.fc2(out5)
            return [out1, out2, out3, out4, out5, out6]
        else:
            return [out5]


class SAD_CNN_Net(nn.Module):
    def __init__(self, output_shape, in_channel=1, mid=1024, one_layer=False):
        super(SAD_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 4 * 1, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not one_layer:
            self.fc2 = nn.Linear(mid, output_shape)

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out4 = out4.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out5 = F.relu(self.fc1(out4))
        if not self.one_layer:
            out6 = self.fc2(out5)
            return [out1, out2, out3, out4, out5, out6]
        else:
            return [out5]

class Image_CNN_Net(nn.Module):
    def __init__(self, output_shape, in_channel=1, mid=1024, one_layer=False):
        super(Image_CNN_Net, self).__init__()
        self.one_layer = one_layer
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp = nn.MaxPool2d(2)
        # fully connect
        # self.fc1 = nn.Linear(64 * 5, 128)
        self.fc1 = nn.Linear(64 * 9 * 9, mid)
        # self.bn3 = nn.BatchNorm1d(128)
        if not one_layer:
            self.fc2 = nn.Linear(mid, output_shape)


    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch
        # x: 64*10*12*12
        out1 = F.relu(self.conv1(x))
        # x: 64*20*4*4
        out2 = F.relu(self.mp(self.conv2(out1)))
        # x: 64*320
        out3 = F.relu(self.conv3(out2))
        out4 = F.relu(self.mp(self.conv4(out3)))

        out5 = F.relu(self.conv5(out4))
        out6 = F.relu(self.mp(self.conv6(out5)))

        out6 = out6.view(in_size, -1) # flatten the tensor
        # x: 64*10
        out7 = F.relu(self.fc1(out6))
        if not self.one_layer:
            out8 = self.fc2(out7)
            return [out1, out2, out3, out4, out5, out6, out7, out8]
        else:
            return [out7]
