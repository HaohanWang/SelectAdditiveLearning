__author__ = 'Haohan Wang'

import numpy as np
import theano
import theano.tensor as T

def shuffleData(data, label, v=None):
    seed = 0
    np.random.seed(seed)
    per = np.random.permutation(data.shape[0])
    data = data[per]
    label = label[per]
    if v is None:
        return data, label
    else:
        v = v[per]
        return data, label, v


def load_data_pretrain(dataset='text'):
    path = '../data/'
    label = np.loadtxt(path + 'MOSI/labels.csv', delimiter=',').astype(int)
    if dataset == 'text':
        data = np.load(path + 'MOSI/textFeatures.npy')
    elif dataset == 'video':
        data = np.load(path + 'MOSI/videoFeatures.npy')
    elif dataset == 'text':
        data = np.load(path + 'MOSI/audioFeatures.npy')
    else:
        data1 = np.load(path + 'MOSI/textFeatures.npy').reshape([label.shape[0],5, 3600]).reshape([label.shape[0], 3600*5])
        data2 = np.load(path + 'MOSI/videoFeatures.npy')
        data3 = np.load(path + 'MOSI/audioFeatures.npy').reshape([label.shape[0],5, 390]).reshape([label.shape[0], 390*5])
        data = np.append(np.append(data1, data2, 1), data3, 1)

    data1 = data[:1250,:]
    label1 = label[:1250]
    data2 = data[1250:,:]
    label2 = label[1250:]

    data1, label1 = shuffleData(data1, label1)

    train_set = (data1[:1000,:], label1[:1000])
    valid_set = (data1[1000:1250,:], label1[1000:1250])

    test_set = (data2, label2)

    print float(sum(train_set[1]))/len(train_set[1]), float(sum(valid_set[1]))/len(valid_set[1]), \
        float(sum(test_set[1]))/len(test_set[1])

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_data_test(dataset='text', t='MOUD'):
    path = '../data/'
    if dataset == 'text':
        data1 = np.load(path + 'MOSI/textFeatures.npy')
        if t == 'MOUD':
            data2 = np.load(path + 'MOUD/textFeatures.npy')
        else:
            data2 = np.load(path + 'YouTube/textFeatures.npy')
    elif dataset == 'video':
        data1 = np.load(path + 'MOSI/videoFeatures.npy')
        if t == 'MOUD':
            data2 = np.load(path + 'MOUD/videoFeatures.npy')
        else:
            data2 = np.load(path + 'YouTube/videoFeatures.npy')
    else:
        data1 = np.load(path + 'MOSI/audioFeatures.npy')
        if t == 'MOUD':
            data2 = np.load(path + 'MOUD/audioFeatures.npy')
        else:
            data2 = np.load(path + 'YouTube/audioFeatures.npy')
    label1 = np.loadtxt(path + 'MOSI/labels.csv', delimiter=',').astype(int)
    if t == 'MOUD':
        label2 = np.loadtxt(path + 'MOUD/labels.csv', delimiter=',').astype(int)
    else:
        label2 = np.loadtxt(path + 'YouTube/labels.csv', delimiter=',').astype(int)

    videoIDs = np.loadtxt(path + 'MOSI/videoIDs.csv', delimiter=',')

    data1, label1, videoIDs = shuffleData(data1, label1, videoIDs)

    train_set = (data1[:1200,:], label1[:1200])
    valid_set = (data1[1200:,:], label1[1200:])
    test_set = (data2, label2)

    train_v = videoIDs[:1200,:]
    test_v = videoIDs[1200:,:]

    return train_set, valid_set, test_set, train_v, test_v

def load_data_self_predict(dataset='text'):
    path = '../data/'
    if dataset == 'text':
        data = np.load(path + 'MOSI/textFeatures.npy')
    elif dataset == 'video':
        data = np.load(path + 'MOSI/videoFeatures.npy')
    else:
        data = np.load(path + 'MOSI/audioFeatures.npy')
    label = np.loadtxt(path + 'MOSI/labels.csv', delimiter=',').astype(int)

    videoIDs = np.loadtxt(path + 'MOSI/videoIDs.csv', delimiter=',')

    data1 = data[:1250,:]
    label1 = label[:1250]
    data2 = data[1250:,:]
    label2 = label[1250:]

    data1, label1, videoIDs = shuffleData(data1, label1, videoIDs)

    train_set = (data1[:1000,:], label1[:1000])
    valid_set = (data1[1000:1250,:], label1[1000:1250])

    train_v = videoIDs[:1000,:]
    test_v = videoIDs[1000:1250,:]

    test_set = (data2, label2)
    return train_set, valid_set, test_set, train_v, test_v