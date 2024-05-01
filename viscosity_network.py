import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k
from keras.callbacks import EarlyStopping
import sklearn.metrics
from sklearn.utils.class_weight import compute_class_weight
from os.path import join
import io

# tf.config.run_functions_eagerly(True)
curr_dir = os.getcwd()
data_file = os.path.join(curr_dir, 'all_data.txt')

all_data = np.loadtxt(data_file)
m = all_data.shape[0]
n = all_data.shape[1]

print(m)
print(n)


mt = int(0.7*m)
mv = m - mt



print(mt)
print(mv)

x = all_data[0:-1, 0:n-1]        # needed in case of using validation split
y = all_data[0:-1, n-1]          # needed when computing class weights 
xt = all_data[0:mt, 0:n-1]
yt = all_data[0:mt, n-1]

xv = all_data[mt:-1, 0:n-1]
yv = all_data[mt:-1, n-1]


# compute class weights


weights = compute_class_weight('balanced', classes = np.unique(y).round(decimals = 2), y=y.round(decimals = 2))
print(weights)
weights = {i: weights[i] for i in range(len(np.unique(y).round(decimals = 2)))}
print(weights)
print(np.unique(y))



# performance metric 'r_square'

def r_square(true, pred):
    res = k.sum(k.square(true-pred))
    total = k.sum(k.square(true - k.mean(true)))
    return(1 - res/(total + k.epsilon()))


# callback/s definition/s

class Trainingplot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        if epoch > 1 and epoch % 50 == 0:  # callback function
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))

            plt.figure(figsize=(10, 6))
            plt.semilogy(N, self.losses, label='Train loss')
            plt.semilogy(N, self.val_losses, label='Validation loss')
      
            plt.title('After epoch = {}'.format(epoch))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('losses.png')

class Predictionsplot(keras.callbacks.Callback):
    def __init__(self, xvp, yvp, model):
        self.model = model
        self.xvp = xvp
        self.yvp = yvp
        self.accs = []
        self.accns = []

    def on_epoch_end(self, epoch, logs={}):
        nzp = self.model.predict(self.xvp)
    
        if epoch > 1 and epoch %10 == 0:
            clear_output(wait=True)
            N = np.arange(len(nzp))
            
            nst = []
            nsp = []
            st = []
            sp = []
            a = len(nzp)
            sum_ns = 0
            sum_s = 0
            for i in range(a):
                if self.yvp[i] == 1.0:
                    nsp.append(nzp[i])
                    nst.append(self.yvp[i])
                    if nzp[i] >= 0.8:
                        sum_ns = sum_ns + 1
                else:
                    st.append(self.yvp[i])
                    sp.append(nzp[i])
                    if nzp[i] <= 0.1:
                        sum_s = sum_s + 1
        


            # plot some labels
            ytrue = self.yvp[0:40]
            xtrue = self.xvp[0:40]
            print(self.xvp)
            pred = self.model.predict(xtrue)
            x = len(ytrue)
            N = np.arange(x)
            for i in range(x):
                if ytrue[i] == 1 and pred[i] <=0.7:
                    print(f'False negatives:{xtrue[i]},label:{ytrue[i]}, pred:{pred[i]}')
                    

            for i in range(x):
                if ytrue[i] == 0 and pred[i] >=0.5:
                    print(f'False positives:{xtrue[i]}, label:{ytrue[i]},pred:{pred[i]}')        

            plt.figure(figsize = (10, 6))
            plt.plot(N, ytrue, 'o', label = 'true')
            plt.plot(N, pred, 'x', label = 'prediction')
            plt.xlabel('ith datapoint')
            plt.ylabel('Labels')
            plt.title(f'Predictions visualization after epoch {epoch}')
            plt.legend()
            plt.savefig('compare.png')
            plt.close()

            accuracy = sum_ns/len(nst)
            self.accns.append(accuracy)        
          

            N_accns = np.arange(0, len(self.accns))
            plt.figure(figsize = (10,6))
            plt.plot(N_accns,self.accns, label = 'Non-smooth accuracy')
            plt.xlabel('Epoch/50')
            plt.title(f'Non-smooth class accuracy after epoch {epoch}: {accuracy}')
            plt.savefig('ns_accuracy.png')
            plt.close()
            

            accuracy = sum_s/len(st)
            self.accs.append(accuracy)
            
            N_accs = np.arange(0, len(self.accs))
            plt.figure(figsize = (10,6))
            plt.plot(N_accs,self.accs, label = 'Smooth accuracy')
            plt.xlabel('Epoch/50')
            plt.title(f'Smooth class Accuracy after epoch {epoch}: {accuracy}')
            plt.savefig('s_accuracy.png')
            plt.close()
            
            plt.figure(figsize = (10, 6))
            predictions = nzp
            bins = np.linspace(0, 1.0, 40)
            plt.hist(predictions, histtype = 'bar', bins = bins)
            plt.xlabel(f'Alpha values predictions after epoch: {epoch}')
            plt.ylabel('Number of alpha values')
            plt.ylim((0, len(self.yvp)))
            plt.legend()
            plt.savefig('pred_histogram.png')
            plt.close()    
            

# neural network architecture

def NN(para):
    opt = para["opt"]
    lrate = para["n"]
    actf = para["A"]
    actp = para["v"]
    cost = para["C"]
    reg = para["reg"]
    regp = para["b"]

    if opt == 'adam':
        opt = keras.optimizers.legacy.Adam(learning_rate = lrate, beta_1=  0.5, beta_2 = 0.99)

    if actf == 'lReLU':
        actf = tf.keras.layers.LeakyReLU(actp)
    
    if actf == 'elu':
        actf = tf.keras.layers.ELU(1.0)
    
    if reg == 'l2':
        reg = keras.regularizers.l2(regp)
    
    if cost == 'mean_squared_error':
        cost = keras.losses.mean_squared_error

    ki = tf.keras.initializers.he_normal  # kernel_initializer
    model = Sequential()
   
    model.add(Dense(10, input_shape=(7,), activation=actf,
                    kernel_regularizer=reg, use_bias=True, kernel_initializer=ki))
    model.add(Dense(10, activation = actf, kernel_regularizer = reg, use_bias = True, kernel_initializer=ki))
    model.add(Dense(10, activation = actf, kernel_regularizer = reg, use_bias = True, kernel_initializer=ki))
    model.add(Dense(10, activation = actf, kernel_regularizer = reg, use_bias = True, kernel_initializer=ki))
    model.add(Dense(10, activation = actf, kernel_regularizer = reg, use_bias = True, kernel_initializer=ki))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = opt, loss = cost, metrics=['mse'])

    return model

# save losses and weights

def save_performance(plot_losses, model, Sb):
    train_losses = plot_losses.losses
    val_losses = plot_losses.val_losses
 

    out_dir = "viscosity_model_performance"
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(join(out_dir, "train_losses_batch_size_{}.txt".format(Sb)), train_losses)
    np.savetxt(join(out_dir, "val_losses_batch_size_{}.txt".format(Sb)), val_losses)
 

    model.save_weights(join(out_dir,'MLP_viscosity_weights_batch_size_{}.h5'.format(Sb)))


# model fit

def trainmlp(model, N, Sb, x, y, n, weights):
    for i in range(n):
        plot_losses = Trainingplot()
        
        plot_predictions = Predictionsplot(xv, yv, model)
        
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 100)
    
        result = model.fit(x, y, batch_size = Sb, epochs = N, verbose = 0, validation_split = 0.3,
                shuffle=True, callbacks=[plot_losses, plot_predictions])
        
        pred = model.predict(xv)
        save_performance(plot_losses, model, Sb)
        
    
        return(pred, result)



para = dict()
para["opt"] = 'adam'
para["n"] = 1e-3                # learning rate
para["A"] = 'lReLU'              # activation function for hidden layers
para["v"] = 1.0e-3               # activation function parameter
para["C"] = 'mean_squared_error' # cost function
para["reg"] = 'l2'               # regularizer
para["b"] = 1.0e-5             # regularization parameter
N = 1000               # maximum number of epochs
Sb = 526                       # mini batch size
n = 3

print(para)


model = NN(para)
pred, result = trainmlp(model, N, Sb, x, y, n, weights)

model.summary()

print("Training loss: ", model.evaluate(xt, yt))
print("Validation loss:", model.evaluate(xv, yv))
print("Accuracy:", model)
print('r_square:', sklearn.metrics.r2_score(yv, pred))



# plot validation labels histogram

plt.figure(figsize = (10, 6))
bins = np.linspace(0, 1.0, 40)
plt.hist(yv, histtype = 'bar', bins = bins)
plt.xlabel('Alpha values')
plt.ylabel('Number of alpha values')
plt.legend()
plt.savefig('validation_histogram.png')


# accuracy analysis

trainp = model.predict(xt)
tp = len(trainp)

valp = model.predict(xv)
vp = len(valp)

sum_one = 0
for i in range(vp):
    if yv[i] == 1 and (valp[i] >= 0.7):
        sum_one = sum_one + 1
  

sum_two = 0
for i in range(tp):
    if (yt[i] == 1) and (trainp[i] >= 0.7):
        sum_two = sum_two + 1
    

print("1s accuracy in training set:", sum_two/np.count_nonzero(yt))
print("1s accuracy in validation set:", sum_one/np.count_nonzero(yv))





