import torch
import numpy as np
import sys
import scipy.linalg as sli
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

def to_tensor(x):
    x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.detach().numpy()

import sklearn.metrics.pairwise as smp
def compute_between_class_distance(data, labels, num_classes=10, eta=1e-3):
    classes = np.unique(labels.reshape([-1]))
    m = np.mean(data, axis=0).reshape([1, -1])
    between_class_distances, within_class_distances = [], []
    Sb, Sw = np.zeros([data.shape[1], data.shape[1]], dtype=np.float32), np.zeros([data.shape[1], data.shape[1]], dtype=np.float32)
    class_centers = []
    for c in classes:
        Xi = data[c == labels].reshape([-1, data.shape[1]])
        mi = np.mean(Xi, axis=0).reshape([1, -1])
        class_centers.append(mi)
        tmp_w = Xi - mi
        Sw += np.dot(tmp_w.T, tmp_w)
        tmp_b = m - mi
        Sb += np.dot(tmp_b.T, tmp_b) * Xi.shape[0]
        between_class_distances.append(np.sqrt(np.sum(tmp_b ** 2)))
        within_class_distances.append(np.mean(np.sqrt(np.sum(tmp_w ** 2, axis=1))))
    class_centers = np.concatenate(class_centers, axis=0)
    min_class_distances = np.sort(smp.pairwise_distances(class_centers), axis=1)[:, 1]
    Sw += np.eye(data.shape[1], dtype=np.float32) * 1e-3
    D, W = sli.eigh(Sb, Sw)
    return within_class_distances, between_class_distances, min_class_distances, np.sort(D)[::-1]


def myLog(x, threshold=100):
    return x.log()


def fisher_loss(data, labels, num_classes=10, eta=1e-3):
    labels_cpu = labels
    if torch.cuda.is_available():
        labels_cpu = labels.cpu()
    classes = torch.unique(labels_cpu.reshape([-1]))
    if torch.cuda.is_available():
        classes = classes.cuda()
    m = data.mean(0).reshape([1, -1])
    Sb, Sw = to_tensor(torch.zeros([data.shape[1], data.shape[1]], dtype=data.dtype)), to_tensor(torch.zeros([data.shape[1], data.shape[1]]))
    for c in classes:
        Xi = data[c == labels].reshape([-1, data.shape[1]])
        mi = Xi.mean(0).reshape([1, -1])
        tmp = Xi - mi
        Sw += tmp.t().mm(tmp)
        tmp = m - mi
        tmp = myActive(tmp)
        Sb += tmp.t().mm(tmp) * Xi.shape[0]

    Sw += to_tensor(torch.eye(data.shape[1], dtype=data.dtype)) * eta
    eigvals = eigh.apply(Sb, Sw)
    eigvals, _ = eigvals.sort(descending=True)
    top_k_evals = eigvals[0: num_classes - 1]
    top_k_evals = top_k_evals[top_k_evals > 0]
    costs = (-myLog(top_k_evals)).mean()
    return costs

def myActive(x):
    return x.sign() * (x.abs() + 1.).log()


def multi_test(data, data_labels, MAP=None, ALL=False):
    n_view = len(data)
    res = np.zeros([n_view, n_view])
    if not ALL:
        if MAP is None:
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    else:
                        from sklearn.neighbors import KNeighborsClassifier
                        neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
                        neigh.fit(data[j], data_labels[j])
                        la = neigh.predict(data[i])
                        res[i, j] = np.sum((la == data_labels[i].reshape([-1])).astype(int)) / float(la.shape[0])
        else:
            if MAP == -1:
                res = [np.zeros([n_view, n_view]), np.zeros([n_view, n_view])]
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    else:
                        if len(data_labels[j].shape) == 1:
                            tmp = fx_calc_map_label(data[j], data_labels[j], data[i], data_labels[i], -1)
                        else:
                            Ks = [50, 0] if MAP == -1 else [MAP]
                            tmp = []
                            for k in Ks:
                                tmp.append(fx_calc_map_multilabel_k(data[j], data_labels[j], data[i], data_labels[i], k=k))
                        if type(tmp) is list:
                            for _i in range(len(tmp)):
                                res[_i][i, j] = tmp[_i]
                        else:
                            res[i, j] = tmp
    else:
        all_data = np.concatenate(data)
        all_labels = np.concatenate(data_labels).reshape([-1])
        if MAP is None:
            for i in range(n_view):
                from sklearn.neighbors import KNeighborsClassifier
                neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
                neigh.fit(all_data, all_labels)
                la = neigh.predict(data[i])
                res[i, 0] = np.sum((la == data_labels[i].reshape([-1])).astype(int)) / float(la.shape[0])
        else:
            if MAP == -1:
                res = [np.zeros([n_view, n_view]), np.zeros([n_view, n_view])]
            for i in range(n_view):
                if len(data_labels[i].shape) == 1:
                    tmp = fx_calc_map_label(all_data, all_labels, data[i], data_labels[i], -1)
                else:
                    Ks = [50, 0] if MAP == -1 else [MAP]
                    tmp = []
                    for k in Ks:
                        tmp.append(fx_calc_map_multilabel_k(all_data, all_labels, data[i], data_labels[i], k=k))
                if type(tmp) is list:
                    for _i in range(len(tmp)):
                        res[_i][i, 0] = tmp[_i]
                else:
                    res[i, 0] = tmp
    return res

import scipy
def fx_calc_map_label(train, train_labels, test, test_label, k=0):
    dist = scipy.spatial.distance.cdist(test, train, 'cosine')
    ord = dist.argsort(1)

    # numcases = dist.shape[1]
    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res


# def fx_calc_map_multilabel_k(image, text, label, k=0, dist_method='L2'):
def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(train, test, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(train, test, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def predict(model, data, batch_size=32, isLong=False):
    batch_count = int(np.ceil(data.shape[0] / float(batch_size)))
    results = []
    with torch.no_grad():
        for i in range(batch_count):
            batch = to_tensor(data[i * batch_size: (i + 1) * batch_size])
            batch = batch.long() if isLong else batch
            results.append(to_data(model(batch)))
            # results.append(to_data(model(batch)))
    return np.concatenate(results)

class eigh(torch.autograd.Function):
    @staticmethod
    def forward(self, Sb, Sw, eigenvectors=False):
        a, b = to_data(Sb), to_data(Sw)
        eta = 0
        for i in range(10):
            try:
                if eta == 0:
                    w, v = sli.eigh(a, b)
                else:
                    w, v = sli.eigh(a, b + np.eye(b.shape[0]) * eta)
                w = w.real.astype('float32')
                v = v.real.astype('float32')
                if eta != 0:
                    Sw += to_tensor(torch.eye(Sw.shape[0]).float()) * eta
                break
            except:
                eta = pow(10, i - 2)

        # w, v = sli.eigh(a, b)
        # w = w.real.astype('float32')
        # v = v.real.astype('float32')

        w, v = to_tensor(w), to_tensor(v)
        self.save_for_backward(Sb, Sw, w, v)
        if eigenvectors:
            return torch.autograd.Variable(w), torch.autograd.Variable(v)
        else:
            return torch.autograd.Variable(w)

    @staticmethod
    def backward(self, grad_output):
        (Sb, Sw, w, v) = self.saved_tensors
        gA = v.mm(torch.diag(grad_output)).mm(v.transpose(1, 0))
        gB = -v.mm(torch.diag(grad_output * w)).mm(v.transpose(1, 0))

        out1 = gA.tril() + gA.triu(1).transpose(1, 0)
        out2 = gB.tril() + gB.triu(1).transpose(1, 0)
        return out1, out2

def show_progressbar(rate, *args, **kwargs):
    '''
    :param rate: [current, total]
    :param args: other show
    '''
    inx = rate[0] + 1
    count = rate[1]
    bar_length = 30
    rate[0] = int(np.around(rate[0] * float(bar_length) / rate[1])) if rate[1] > bar_length else rate[0]
    rate[1] = bar_length if rate[1] > bar_length else rate[1]
    num = len(str(count))
    str_show = ('\r%' + str(num) + 'd / ' + '%' + str(num) + 'd  (%' + '3.2f%%) [') % (inx, count, float(inx) / count * 100)
    for i in range(rate[0]):
        str_show += '='

    if rate[0] < rate[1] - 1:
        str_show += '>'

    for i in range(rate[0], rate[1] - 1, 1):
        str_show += '.'
    str_show += '] '
    for l in args:
        str_show += ' ' + str(l)

    for key in kwargs:
        try:
            str_show += ' ' + key + ': %.4f' % kwargs[key]
        except Exception:
            str_show += ' ' + key + ': ' + str(kwargs[key])
    if inx == count:
        str_show += '\n'

    sys.stdout.write(str_show)
    sys.stdout.flush()