import numpy as np
from sklearn.datasets import fetch_mldata
# from sklearn.datasets import fetch_openml
from chainer.datasets import TupleDataset
import pickle
from chainer import serializers
import pandas

def load_dataset(data_id, n_p, n_n, n_u, prior, n_t, n_vp=None, n_vn=None, n_vu=None):
    if data_id == 0:
        data_name = "MNIST"
        x, y = get_mnist()
        y = y.astype(int)
        x, y = x/255., binarize_mnist(y)
        pos, neg = +1, -1
    elif data_id == 1:
        data_name = "Nichschraing_TFIDF_chars"
        prior = prior / 10

        xtrain = pickle.load(open("./data/xtrain_tfidf_ngram_chars.obj","rb"))
        xtrain_arr = np.zeros(xtrain.shape)
        xtrain = xtrain.todense(out=xtrain_arr)

        xval = pickle.load(open("./data/xvalid_tfidf_ngram_chars.obj","rb"))
        xval_arr = np.zeros(xval.shape)
        xval = xval.todense(out=xval_arr)

        xtrain, xval = xtrain_arr, xval_arr
        print("x shapes train, val", xtrain.shape, xval.shape)

        trainY = pickle.load(open("./data/trainY.obj", "rb"))
        validY = pickle.load(open("./data/validY.obj", "rb"))
        print("y shapes train, val", trainY.shape, validY.shape)

        #x = np.r_[xtrain, xval]
        #y = np.r_[trainY, validY]
        x = np.append(xtrain, xval, axis=0)
        y = np.append(trainY, validY, axis=0)
        pos, neg = 1, -1
        y[y == 0] = neg
    data_p, data_n = x[y == pos, :], x[y == neg, :]
    print("data_p, data_n", data_p.shape, data_n.shape)
    n_up, n_un = split_size(n_u, prior)

    data_p, data_n, x_l, y_l = split_data(data_p, data_n, n_p, n_n)
    data_p, data_n, x_u, y_u = split_data(data_p, data_n, n_up, n_un)
    if n_vp is not None and n_vn is not None:
        data_p, data_n, x_vl, y_vl = split_data(data_p, data_n, n_vp, n_vn)
    if n_vu is not None:
        n_vup, n_vun = split_size(n_vu, prior)
        data_p, data_n, x_vu, y_vu = split_data(data_p, data_n, n_vup, n_vun)
    data_p, data_n, x_t, y_t = split_data(data_p, data_n, n_t, n_t)

    x_p, x_n = x_l[y_l == +1, :], x_l[y_l == -1, :]
    if n_vp is not None and n_vn is not None and n_vu is not None:
        x_vp, x_vn = x_vl[y_vl == +1, :], x_vl[y_vl == -1, :]
        return data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn, x_vu, y_vu
    if n_vp is not None and n_vn is not None:
        return data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn
    return data_name, x_p, x_n, x_u, y_u, x_t, y_t



def get_mnist():
    mnist = fetch_mldata('MNIST original', data_home='~') # fetch_mldata deprecated
    # mnist = fetch_openml('mnist_784', version=1, cache=True, data_home='~')
    x, y = mnist.data, mnist.target
    return x, y

def binarize_mnist(org_y):
    y = np.ones(len(org_y))
    # print(org_y)
    y[org_y % 2 == 1] = -1
    return y

def split_data(data_p, data_n, n_p, n_n):
    N_p, N_n = data_p.shape[0], data_n.shape[0]
    if (n_p == -1): n_p = N_p
    if (n_n == -1): n_n = N_n
    index_p, index_n = split_index(N_p, n_p), split_index(N_n, n_n)
    x_up, x_un = data_p[index_p, :], data_n[index_n, :]
    data_p, data_n = data_p[np.logical_not(index_p), :], data_n[np.logical_not(index_n), :]
    x, y = np.r_[x_up, x_un], np.r_[np.ones(n_p), -np.ones(n_n)]
    return data_p, data_n, x, y

def split_size(n, prior):
    n_p = np.random.binomial(n, prior)
    n_n = n - n_p
    return n_p, n_n
    
def split_index(N, n):
    if n > N:
        raise Exception("""n > N {0} > {1} The number of samples is small.
Use large-scale dataset or reduce the size of training data.
""".format(n, N))

    index = np.zeros(N, dtype=bool)
    index[np.random.permutation(N)[:n]] = True
    return index

if __name__ == "__main__":
    import pickle
    data_id, prior = 0, .5
    n_p, n_n, n_u, n_t, n_vp, n_vn, n_vu = 100, 0, 10000, 100, 20, 20, 100
    data_id, prior = 1, 0.05
    n_p, n_n, n_u, n_t, n_vp, n_vn, n_vu = 1500, 0, 16000, -1, 500, 150, 2000
    data_name, x_p, x_n, x_u, y_u, x_t, y_t, x_vp, x_vn, x_vu, y_vu \
    = load_dataset(data_id, n_p, n_n, n_u, prior, n_t, n_vp=n_vp, n_vn=n_vn, n_vu=n_vu)
    exit(0)
    pickle.dump(data_name, open('MNIST_data_name.obj', 'wb+'))
    pickle.dump(x_p, open('MNIST_x_p.obj', 'wb+'))
    pickle.dump(x_n, open('MNIST_x_n.obj', 'wb+'))
    pickle.dump(x_u, open('MNIST_x_u.obj', 'wb+'))
    pickle.dump(y_u, open('MNIST_y_u.obj', 'wb+'))
    pickle.dump(x_t, open('MNIST_x_t.obj', 'wb+'))
    pickle.dump(y_t, open('MNIST_y_t.obj', 'wb+'))
    pickle.dump(x_vp, open('MNIST_x_vp.obj', 'wb+'))
    pickle.dump(x_vn, open('MNIST_x_vn.obj', 'wb+'))
    pickle.dump(x_vu, open('MNIST_x_vu.obj', 'wb+'))
