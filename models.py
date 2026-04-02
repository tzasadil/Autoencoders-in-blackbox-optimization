
import sys
import VAE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import sklearn.gaussian_process.kernels as GPK
import progress_bar
import math
from rbf_layer import InitCentersRandom, RBFLayer
from functools import partial as p
from functools import partial
import VAE
from rbf_layer import RBFLayer
import scipy





#dimensionality reductions
def id(x,y=None,w=None,model=None): 
    return lambda a:a

def pca(bottleneck,x,model):
    inp_dim = x.shape[-1]
    pca_dim = int(bottleneck*inp_dim) if isinstance(bottleneck,float) else bottleneck
    pca_dim = min(pca_dim,x.shape[0])
    pca = PCA(pca_dim).fit(x)
    return pca.transform

def vae(l,x,model):
    d = x.shape[-1]
    if model == None:
        model= VAE.VAE(d,l)
        model.compile()
    model.fit(x,x,batch_size = max(1, int(x.shape[0]/10)),epochs=2,verbose=0)
    return model


# predictors

def gp(kernel,x,y,model):
    # 'custom' optimizer just to set a different maxiter
    def opt(obj_func,initial_theta,bounds): 
        res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={'maxiter':1000},
            )
        return res.x,res.fun
    gp =  GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=5,optimizer=opt)
    gp.fit(x, y)
    return gp.predict

    
def elm(h,x,y,model):
    inp_size = x.shape[-1]
    hidden_size = int(h*inp_size)
    input_weights = tf.random.normal([inp_size,hidden_size])
    biases = tf.random.normal([hidden_size])
    h = lambda a: tf.nn.relu(tf.tensordot(a,input_weights,1) + biases)
    # h = lambda a: tf.nn.relu(tf.tensordot(a,input_weights,1) + biases)
    output_weights = tf.tensordot(tf.linalg.pinv(h(tf.cast(x,tf.float32))), tf.cast(y,tf.float32),1)
    inp = tf.keras.layers.Input(shape=(inp_size,))
    outp = tf.tensordot(h(inp),output_weights,1)
    model = tf.keras.Model(inputs=inp,outputs=outp)
    return model

def rbf_network(layers,gamma,x,y,model): 
    d = x.shape[-1]
    layers = [int(d*n) for n in layers]
    if model == None or model.layers[0].input_shape[0][-1] != d:
        inp = tf.keras.layers.Input(shape=(d,))
        feed = inp
        for n in layers:
            feed = RBFLayer(n,InitCentersRandom(x), gamma)(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            # feed = tf.nn.relu(feed)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        outp = tf.keras.layers.Dense(1)(feed)
        outp = tf.squeeze(outp,-1)
        model = tf.keras.Model(inputs=inp,outputs=outp)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/20),epochs=3,verbose=0)
    return model

def mlp(layers,x,y,model):
    d = x.shape[-1]
    layers = [int (d*n) for n in layers]
    if model == None or model.layers[0].input_shape[0][-1] != d:
        inp = tf.keras.layers.Input(shape=(d,))
        feed = inp
        for n in map(int,layers):
            feed = tf.keras.layers.Dense(n)(feed)# + (int(feed.shape[-1] == n) * feed if feed.shape[-1] == n else 0)
            feed = tf.nn.relu(feed)
            # feed = tf.keras.layers.Dropout(0.2)(feed)
        outp = tf.keras.layers.Dense(1)(feed)
        outp = tf.squeeze(outp,-1)
        model = tf.keras.Model(inputs=inp,outputs=outp)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss = 'mse')
    model.fit(x,y,batch_size = int(x.shape[0]/10),epochs=5,verbose=0)
    return model

def nearest(k,x,y,model):
    def call(arr):
        res = []
        for a in arr:
            eu_dist = np.linalg.norm(x - a.reshape([1,-1]), axis=1)
            dists_i = np.argsort(eu_dist)
            res.append(np.mean([y[i] for i in dists_i[:k]]))
        return np.array(res)
    return call




# class Ansamble :
#     @staticmethod 
#     def create(combination_f, models):
#         listmap = lambda func, collection: list(map(func, collection))
#         self = Ansamble()
#         self.combination_f = combination_f
#         self.model_fs = listmap(lambda a: a[0], models)
#         self.model_descs = listmap(lambda a: a[1], models)
#         self.old_models = listmap(lambda _: None, self.model_fs)
#         return (
#             self,
#             'ansamble_[' + '&'.join(self.model_descs) + ']'
#         )
#     def __call__(self, data):
#             called = [m(data) for m in self.models]
#             stacked = np.stack(called,0),
#             combined = self.combination_f(stacked, axis=0)
#             return combined 
#     def train(self,x,y):
#         self.models = [f(h,x,y,m_old) for (f,m_old) in zip(self.model_fs,self.models)]




class Surrogate:
    def __init__(self,model_f,dim_red_f, min_pop=0, train_records=200):
        self.is_id = model_f is None and dim_red_f is None
        self.model_f = model_f if model_f != None else lambda x,y,m: lambda a: a[:,0]
        self.dim_red_f = dim_red_f if dim_red_f != None else lambda x,m: lambda a: a
        self.train_records = train_records
        self.model = None
        self.dim_red = None
        self.min_pop = min_pop
        self.inp_size = min_pop
    def __call__(self,x):
            if self.is_id: return np.zeros(x.shape[0])
            latent = self.dim_red(x)  # ty:ignore[call-non-callable]
            y = self.model(latent)  # ty:ignore[call-non-callable]
            y = np.nan_to_num(y, nan=4.9)
            return y
    
    def train(self, train_x, train_y,opt=None):
        if self.is_id: return
        X = np.array(train_x)
        Y = np.array(train_y) 
    
        xx,yy = X, Y
        if self.train_records is not None: 
            xx = xx[-self.train_records:]
            yy = yy[-self.train_records:]


            # yy = np.round(np.array(yy), decimals=3)
            # yy_, inx = np.unique(yy, return_index=True)
            # xx_ = np.array(xx)[inx]
            # if len(inx)>=self.inp_size:
            #     xx, yy = xx_, yy_
            # eu_dist = np.linalg.norm(xx - opt._mean.reshape([1,-1]), axis=1)
            # dists_i = np.argsort(eu_dist)[:self.train_records]
            # xx = xx[dists_i]
            # yy = yy[dists_i]

        self.dim_red = self.dim_red_f(xx,self.dim_red)
        latentX = self.dim_red(xx)
        self.model = self.model_f(latentX,yy,self.model)
        
        


if __name__ == '__main__':
    import main
    main.main()