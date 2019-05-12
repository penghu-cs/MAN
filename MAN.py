import torch
import numpy as np

from torch import optim
import utils_PyTorch as utils
import data_loader
from torch.autograd import Variable
import scipy.io as sio

class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        wv_matrix = None

        self.output_shape = config.output_shape
        data = data_loader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        if len(data) == 8:
            (self.train_data, self.train_labels, self.valid_data, self.valid_labels, self.test_data, self.test_labels, wv_matrix, self.MAP) = data
            self.text_views = [len(self.train_data) - 1]
        elif len(data) == 7:
            (self.train_data, self.train_labels, self.valid_data, self.valid_labels, self.test_data, self.test_labels, self.MAP) = data
            self.text_views = [-1]
        elif len(data) == 6:
            (self.train_data, self.train_labels, self.test_data, self.test_labels, wv_matrix, self.MAP) = data
            self.text_views = [len(self.train_data) - 1]
        elif len(data) == 5:
            (self.train_data, self.train_labels, self.test_data, self.test_labels, self.MAP) = data
            self.text_views = [-1]
        self.n_view = len(self.train_data)
        self.num_classes = len(np.unique(np.concatenate(self.train_labels).reshape([-1]))) if len(self.train_labels[0].shape) == 1 else self.train_labels[0].shape[1]
        if self.output_shape == -1:
            self.output_shape = self.num_classes


        self.mode = config.text_mode
        self.word_dim = 300
        self.dropout_prob = 0.5
        if wv_matrix is not None:
            self.vocab_size = wv_matrix.shape[0] - 2


        self.filters = [3, 4, 5]
        # self.filter_num = [64, 64, 64]
        self.filter_num = [100, 100, 100]
        self.ALL = config.ALL

        if self.mode != 'rand':
            print("loading word2vec...")
            self.wv_matrix = wv_matrix
            self.word_dim = self.wv_matrix.shape[1] if wv_matrix is not None else 0

        self.input_shape = [self.train_data[v].shape[1] for v in range(self.n_view)]

        self.g12 = None
        self.g21 = None
        self.D = None
        self.train_view_list = [[] for v in range(self.n_view)]
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.lr = config.lr

        self.epochs = config.epochs
        self.sample_interval = config.sample_interval
        self.eta = config.eta
        self.compute_all = config.compute_all
        self.fisher_beta = config.fisher_beta
        self.just_valid = config.just_valid
        if self.batch_size < 0:
            self.batch_size = 100 if self.num_classes < 100 else 500

        self.criterion_class = lambda x, y: ((x - y) ** 2).sum(1).mean() / 2.
        self.criterion_view = lambda x, y: ((x - y) ** 2).sum(1).mean() / 2.

        self.build_model()

        self.len_list = [d.shape[0] for d in self.train_data]


    def build_model(self):
        """Builds a generator and a discriminator."""
        from model import Dense_Net, Text_CNN_list, D
        self.Gs = []
        for i in range(self.n_view):
            if i in self.text_views:
                self.Gs.append(Text_CNN_list(mode=self.mode, word_dim=self.word_dim, vocab_size=self.vocab_size, out_dim=self.output_shape, filters=self.filters, filter_num=self.filter_num, dropout_prob=self.dropout_prob, wv_matrix=self.wv_matrix))
            else:
                self.Gs.append(Dense_Net(input_dim=self.input_shape[i], out_dim=self.output_shape))
        self.D = D(dim=self.output_shape, view=self.n_view)
        get_grad_params = lambda model: [x for x in model.parameters() if x.requires_grad]
        g_params = [params for G in self.Gs for params in get_grad_params(G)]
        d_params = get_grad_params(self.D)
        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        if torch.cuda.is_available():
            for G in self.Gs:
                G.cuda()
            self.D.cuda()

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x) #torch.autograd.Variable

    def to_data(self, x):
        """Converts variable to numpy."""
        try:
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy()
        except Exception as e:
            return x

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def view_result(self, _acc):
        res = ''
        if type(_acc) is not list:
            _acc = _acc[_acc.nonzero()]
            res += ((' - mean: %.5f' % _acc.mean()) + ' - detail:')
            for _i in range(len(_acc)):
                res += ('%.5f' % _acc[_i]) + ','
        else:
            R = [50, 'ALL']
            for _k in range(len(_acc)):
                _acc_t = _acc[_k]
                _acc_t = _acc_t[_acc_t.nonzero()]

                res += (' R = ' + str(R[_k]) + ': ')
                res += ((' - mean: %.5f' % _acc_t.mean()) + ' - detail:')
                for _i in range(len(_acc_t)):
                    res += ('%.5f' % _acc_t[_i]) + ','
        return res

    def shuffleInx(self):
        inx_list = []
        for v in range(self.n_view):
            length = self.len_list[v]
            if length in self.len_list[0: v]:
                inx_list.append(inx_list[self.len_list.index(length)])
            else:
                rand_idx = np.arange(length)
                np.random.shuffle(rand_idx)
                inx_list.append(rand_idx)
        return inx_list

    def train(self):
        # valid, fake_1 = np.ones([self.batch_size, 1]), np.zeros([self.batch_size, 1])
        max_sum_results, best_results = 0., 0.
        discriminator_losses, generator_losses, fisher_losses, valid_results = [], [], [], []

        train_features_list, valid_features_list, test_features_list = [], [], []
        if self.just_valid:
            train_features_list_tmp, valid_features_list_tmp, test_features_list_tmp = [], [], []
            for v in range(self.n_view):
                train_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.train_data[v], self.batch_size).reshape([self.train_data[v].shape[0], -1]))
                valid_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.valid_data[v], self.batch_size).reshape([self.valid_data[v].shape[0], -1]))
                test_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.test_data[v], self.batch_size).reshape([self.test_data[v].shape[0], -1]))
            train_features_list.append([train_features_list_tmp, self.train_labels])
            valid_features_list.append([valid_features_list_tmp, self.valid_labels])
            test_features_list.append([test_features_list_tmp, self.test_labels])
        for epoch in range(self.epochs):
            print(('Epoch %d/%d') % (epoch + 1, self.epochs))
            rand_idx_list = self.shuffleInx()

            batch_count = int(np.ceil(min(self.len_list) / float(self.batch_size)))

            k = 0
            mean_d_real_loss, mean_d_loss, mean_d_fake_loss, mean_g_loss, mean_f_loss = [], [], [], [], []
            for batch_idx in range(batch_count):
                # idx1 = rand_idx1[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

                train_x, train_y, view_y, view_y_fake, train_dist_map = [], [], [], [], []
                for i in range(self.n_view):
                    idx = rand_idx_list[i][batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
                    train_x.append(self.to_var(torch.tensor(self.train_data[i][idx])))
                    if len(self.train_labels[i].shape) == 1 or self.train_labels[i].shape[1] == 1:
                        train_y.append(self.to_var(torch.tensor(self.train_labels[i][idx])))
                    else:
                        train_y.append(self.to_var(torch.tensor(self.train_labels[i][idx].astype('float32'))))

                    one_hot = torch.FloatTensor(idx.shape[0], self.n_view).float()
                    one_hot.zero_()
                    one_hot.scatter_(1, torch.tensor([i] * train_y[i].shape[0]).reshape([-1, 1]), 1)
                    view_y.append(self.to_var(one_hot))

                    one_hot = torch.FloatTensor(idx.shape[0], self.n_view).float()
                    one_hot.zero_()
                    one_hot.scatter_(1, torch.tensor([i] * train_y[i].shape[0]).reshape([-1, 1]), 1)
                    view_y_fake.append(self.to_var((one_hot != 1).float() / (self.n_view - 1)))

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.reset_grad()
                d_loss = 0.
                for v in range(self.n_view):
                    gc = self.Gs[v](train_x[v])
                    view_out = self.D(gc[-1])
                    d_loss += self.criterion_view(view_out, view_y[v])
                d_loss.backward()
                self.d_optimizer.step()

                # ---------------------
                #  Train Generators
                # ---------------------
                self.reset_grad()
                g_loss = 0.
                gc_list = []
                for v in range(self.n_view):
                    gc = self.Gs[v](train_x[v])
                    gc_list.append(gc[-1])
                    view_out = self.D(gc[-1])
                    g_loss += self.criterion_view(view_out, view_y_fake[v])
                fisher_loss = utils.fisher_loss(torch.cat(gc_list), torch.cat(train_y), num_classes=self.num_classes, eta=self.eta)
                loss = g_loss + fisher_loss * self.fisher_beta
                loss.backward()
                self.g_optimizer.step()

                mean_d_loss.append(self.to_data(d_loss))
                mean_g_loss.append(self.to_data(g_loss))
                mean_f_loss.append(self.to_data(fisher_loss))
                if ((epoch + 1) % self.sample_interval == 0) and (batch_idx == batch_count - 1):
                    valid_pre, train_pre, d_results = [], [], []
                    for v in range(self.n_view):
                        try:
                            valid_pre.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.valid_data[v], self.batch_size).reshape([self.valid_data[v].shape[0], -1]))
                        except Exception:
                            pass
                        train_pre.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.train_data[v], self.batch_size).reshape([self.train_data[v].shape[0], -1]))
                        d_results.append((utils.predict(lambda x: self.D(x), train_pre[v]).argmax(axis=1) == v).sum() / float(self.train_labels[v].shape[0]))

                    if len(valid_pre) > 0:
                        _val_result = utils.multi_test(valid_pre, self.valid_labels, self.MAP, self.ALL)
                    else:
                        test_pre = []
                        for v in range(self.n_view):
                            test_pre.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.test_data[v], self.batch_size).reshape([self.test_data[v].shape[0], -1]))
                        _val_result = utils.multi_test(test_pre, self.test_labels, self.MAP, self.ALL)


                    cross_view_str = 'valid-acc: ' + self.view_result(_val_result)
                    tmp_resutls = _val_result[-1] if type(_val_result) is list else _val_result
                    if self.compute_all or np.sum(max_sum_results) < np.sum(tmp_resutls):
                        max_sum_results = tmp_resutls
                        best_valid_results = _val_result

                        if not self.just_valid:
                            if len(valid_pre) > 0:
                                test_pre = []
                                for v in range(self.n_view):
                                    test_pre.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.test_data[v], self.batch_size).reshape([self.test_data[v].shape[0], -1]))

                                tmp = best_results
                                best_results = utils.multi_test(test_pre, self.test_labels, self.MAP, self.ALL)
                                cross_view_str += '\ttest-acc: ' + self.view_result(best_results)
                                best_results = tmp if self.compute_all and np.sum(tmp) > np.sum(best_results) else best_results
                                sio.savemat('feature_results/' + self.datasets + '_ALL_' + str(self.ALL) + '_test_feature_results.mat', {'data': np.array(test_pre), 'labels': np.array(self.test_labels)})
                        best_epoch = epoch

                    if self.just_valid and (epoch + 1) % 50 == 0:
                        train_features_list_tmp, valid_features_list_tmp, test_features_list_tmp = [], [], []
                        for v in range(self.n_view):
                            train_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.train_data[v], self.batch_size).reshape([self.train_data[v].shape[0], -1]))
                            valid_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.valid_data[v], self.batch_size).reshape([self.valid_data[v].shape[0], -1]))
                            test_features_list_tmp.append(utils.predict(lambda x: self.Gs[v](x)[-1].view([x.shape[0], -1]), self.test_data[v], self.batch_size).reshape([self.test_data[v].shape[0], -1]))
                        train_features_list.append([train_features_list_tmp, self.train_labels])
                        valid_features_list.append([valid_features_list_tmp, self.valid_labels])
                        test_features_list.append([test_features_list_tmp, self.test_labels])
                    utils.show_progressbar([batch_idx, batch_count], mean_D_loss=np.mean(mean_d_loss), mean_G_loss=np.mean(mean_g_loss), mean_fisher_loss=np.mean(mean_f_loss), D_view_acc=np.mean(d_results),  cross_view_acc=cross_view_str)

                    discriminator_losses.append(np.mean(mean_d_loss))
                    generator_losses.append(np.mean(mean_g_loss))
                    fisher_losses.append(np.mean(mean_f_loss))
                    valid_results.append(_val_result)

                elif batch_idx == batch_count - 1:
                    utils.show_progressbar([batch_idx, batch_count], mean_D_loss=np.mean(mean_d_loss), mean_G_loss=np.mean(mean_g_loss), mean_f_loss=np.mean(mean_f_loss))
                else:
                    utils.show_progressbar([batch_idx, batch_count], D_loss=d_loss, G_loss=g_loss, F_loss=fisher_loss)
                k += 1
        if self.just_valid:
            print("best_epoch: %d" % best_epoch + ",\t valid best resutls:" + self.view_result(best_valid_results))
            return best_valid_results, discriminator_losses, generator_losses, fisher_losses, valid_results, train_features_list, valid_features_list, test_features_list
        else:
            print("best_epoch: %d" % best_epoch + ",\t valid best resutls:" + self.view_result(best_valid_results) + ",\t best resutls:" + self.view_result(best_results))
            return best_valid_results, best_results
