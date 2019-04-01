import h5py
import re
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import scipy.io as sio

def load_deep_features(data_name):
    import h5py
    valid_data, req_rec, b_wv_matrix = True, True, True
    if data_name == 'wiki':
        path = './datasets/wiki_deep_idx_data-corr-ae.h5py'
        MAP = -1
    elif data_name == 'pascal':
        path = './datasets/pascal_deep_idx_data-corr-ae.h5py'
        MAP = -1
    elif data_name == 'xmedia':
        path = './datasets/XMediaFeatures.mat'
        MAP = -1
        req_rec, b_wv_matrix = False, False
        all_data = sio.loadmat(path)
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature
        I_te = all_data['I_te'].astype('float32')           # Features of test set for image data, BoVW feature
        I_tr = all_data['I_tr'].astype('float32')           # Features of training set for image data, BoVW feature
        T_te = all_data['T_te'].astype('float32')           # Features of test set for text data, LDA feature
        T_tr = all_data['T_tr'].astype('float32')           # Features of training set for text data, LDA feature
        V_te = all_data['V_te'].astype('float32')           # Features of test set for video(frame) data, BoVW feature
        V_tr = all_data['V_tr'].astype('float32')           # Features of training set for video(frame) data, BoVW feature
        I_te_CNN = all_data['I_te_CNN'].astype('float32')	# Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')	# Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')	# Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')	# Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')	# Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')	# Features of training set for video(frame) data, CNN feature
        te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64') - 1   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64') - 1   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64') - 1 # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64') - 1 # category label of training set for audio data
        teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64') - 1 # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64') - 1 # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64') - 1 # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1]).astype('int64') - 1 # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64') - 1 # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64') - 1 # category label of training set for text data

        train_data = [A_tr, d3_tr, T_tr_BOW, I_tr_CNN, V_tr_CNN]
        valid_data = [A_te, d3_te, T_te_BOW, I_te_CNN, V_te_CNN]
        test_data = [A_te, d3_te, T_te_BOW, I_te_CNN, V_te_CNN]
        train_labels = [trAudCat, tr3dCat, trTxtCat, trImgCat, trVidCat]
        valid_labels = [teAudCat, te3dCat, teTxtCat, teImgCat, teVidCat]
        test_labels = [teAudCat, te3dCat, teTxtCat, teImgCat, teVidCat]
    elif data_name == 'xmedia_pairwise':
        path = './datasets/XMediaFeatures_Pairwise.mat'
        MAP = -1
        req_rec, b_wv_matrix = False, False
        all_data = sio.loadmat(path)
        A_te = all_data['A_te'].astype('float32')           # Features of test set for audio data, MFCC feature
        A_tr = all_data['A_tr'].astype('float32')           # Features of training set for audio data, MFCC feature
        d3_te = all_data['d3_te'].astype('float32')         # Features of test set for 3D data, LightField feature
        d3_tr = all_data['d3_tr'].astype('float32')         # Features of training set for 3D data, LightField feature

        I_te_CNN = all_data['I_te_CNN'].astype('float32')	# Features of test set for image data, CNN feature
        I_tr_CNN = all_data['I_tr_CNN'].astype('float32')	# Features of training set for image data, CNN feature
        T_te_BOW = all_data['T_te_BOW'].astype('float32')	# Features of test set for text data, BOW feature
        T_tr_BOW = all_data['T_tr_BOW'].astype('float32')	# Features of training set for text data, BOW feature
        V_te_CNN = all_data['V_te_CNN'].astype('float32')	# Features of test set for video(frame) data, CNN feature
        V_tr_CNN = all_data['V_tr_CNN'].astype('float32')	# Features of training set for video(frame) data, CNN feature
        te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64') - 1   # category label of test set for 3D data
        tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64') - 1   # category label of training set for 3D data
        teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64') - 1 # category label of test set for audio data
        trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64') - 1 # category label of training set for audio data
        teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64') - 1 # category label of test set for image data
        trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64') - 1 # category label of training set for image data
        teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64') - 1 # category label of test set for video(frame) data
        trVidCat = all_data['trVidCat'].reshape([-1]).astype('int64') - 1 # category label of training set for video(frame) data
        teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64') - 1 # category label of test set for text data
        trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64') - 1 # category label of training set for text data

        train_data = [A_tr, d3_tr, T_tr_BOW, I_tr_CNN, V_tr_CNN]
        valid_data = [A_te, d3_te, T_te_BOW, I_te_CNN, V_te_CNN]
        test_data = [A_te, d3_te, T_te_BOW, I_te_CNN, V_te_CNN]
        train_labels = [trAudCat, tr3dCat, trTxtCat, trImgCat, trVidCat]
        valid_labels = [teAudCat, te3dCat, teTxtCat, teImgCat, teVidCat]
        test_labels = [teAudCat, te3dCat, teTxtCat, teImgCat, teVidCat]

    if req_rec:
        h = h5py.File(path)
        train_imgs_deep = h['train_imgs_deep'][()].astype('float32')
        train_imgs_labels = h['train_imgs_labels'][()]
        train_imgs_labels -= np.min(train_imgs_labels)
        train_texts_idx = h['train_texts_idx'][()].astype('int64')
        train_texts_labels = h['train_texts_labels'][()]
        train_texts_labels -= np.min(train_texts_labels)
        train_data = [train_imgs_deep, train_texts_idx]
        train_labels = [train_imgs_labels, train_texts_labels]

        try:
            valid_imgs_deep = h['valid_imgs_deep'][()].astype('float32')
            valid_imgs_labels = h['valid_imgs_labels'][()]
            valid_imgs_labels -= np.min(valid_imgs_labels)
            valid_texts_idx = h['valid_texts_idx'][()].astype('int64')
            valid_texts_labels = h['valid_texts_labels'][()]
            valid_texts_labels -= np.min(valid_texts_labels)
            valid_data = [valid_imgs_deep, valid_texts_idx]
            valid_labels = [valid_imgs_labels, valid_texts_labels]
        except Exception as e:
            valid_data = False

        test_imgs_deep = h['test_imgs_deep'][()].astype('float32')
        test_imgs_labels = h['test_imgs_labels'][()]
        test_imgs_labels -= np.min(test_imgs_labels)
        test_texts_idx = h['test_texts_idx'][()].astype('int64')
        test_texts_labels = h['test_texts_labels'][()]
        test_texts_labels -= np.min(test_texts_labels)
        wv_matrix = h['wv_matrix'][()] if b_wv_matrix else None
        test_data = [test_imgs_deep, test_texts_idx]
        test_labels = [test_imgs_labels, test_texts_labels]
    else:
        pass

    if valid_data:
        if b_wv_matrix:
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, wv_matrix, MAP
        else:
            return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, MAP
    else:
        if b_wv_matrix:
            return train_data, train_labels, test_data, test_labels, wv_matrix, MAP
        else:
            return train_data, train_labels, test_data, test_labels, MAP
