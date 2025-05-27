import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt 
''' LEARNING RATE SCHEDULERS'''

# plot_lr = False
# if plot_lr:
#     lr = [1e-3]
#     for i in range(100):
#         lr.append(lr_exp(i,lr[-1]))
#     plt.plot(lr)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def lr_exp(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def choose_optimizer_scheduler(optimizer,lr_sched,learning_rate):
    lr_scheduler = choose_lr_scheduler(lr_sched)
    if lr_sched == 'None' or lr_sched == 'plateau':
        optimizer = choose_optimizer(name_optimizer=optimizer,lr=learning_rate)
    else: 
        optimizer = choose_optimizer_with_scheduler(optimizer, lr_scheduler)
    return optimizer,lr_scheduler


def choose_lr_scheduler(name_lr_scheduler=None):
    if name_lr_scheduler == 'exponential_decay':
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.1,decay_steps=30,decay_rate=0.01)
    elif name_lr_scheduler == 'cosine_decay':
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'cosine_decay_restarts':
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.1,first_decay_steps=5)
    elif name_lr_scheduler == 'inverse_time_decay':
        lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=.1,decay_steps=1,decay_rate = 0.5)
    elif name_lr_scheduler == 'piece_wise_constant_decay':
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'polynomial_decay':
        lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'plateau':
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-4,mode='min') 
        # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_task_mrr', factor=0.5, patience=10, min_lr=1e-4,mode='max') 
    elif name_lr_scheduler == 'custom':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr) # lr_exp function
    elif name_lr_scheduler == 'cyclical':
        steps_per_epoch = 2
        lr_scheduler = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            scale_fn=lambda x: 1/(2.**(x-1)),
            step_size=2 * steps_per_epoch
        )
    else: 
        print('No lr_scheduler chosen!')
        lr_scheduler=None
    # step = np.arange(0, 200)
    # if (name_lr_scheduler!='plateau') and (name_lr_scheduler != 'custom'):
    #     lr = lr_scheduler(step)
    #     plt.plot(step, lr)
    #     plt.xlabel("Steps")
    #     plt.ylabel("Learning Rate")
    #     plt.show()
    return lr_scheduler



''' OPTIMIZERS '''


def choose_optimizer(name_optimizer='adam',lr=0.001):
    if name_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif name_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif name_optimizer == 'adafactor':
        optimizer = tf.keras.optimizers.Adafactor(learning_rate=lr)
    elif name_optimizer == 'nadam':
        optimizer = tf.keras.optimizers.experimental.Nadam(learning_rate=lr)
    elif name_optimizer == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=lr)
    elif name_optimizer == 'ftrl':
        optimizer = tf.keras.optimizers.experimental.Ftrl(learning_rate=lr)
    elif name_optimizer == 'adamax':
        optimizer = tf.keras.optimizers.experimental.Adamax(learning_rate=lr)
    elif name_optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.experimental.Adagrad(learning_rate=lr)
    elif name_optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.experimental.Adadelta(learning_rate=lr)
    else: 
        print('Name of optimizer not valid!')
        optimizer=None
    return optimizer


def choose_optimizer_with_scheduler(name_optimizer, lr_scheduler):
    if name_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
    if name_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    elif name_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.RMSprop(lr_scheduler)
    elif name_optimizer == 'adafactor':
        optimizer = tf.keras.optimizers.Adafactor(lr_scheduler)
    elif name_optimizer == 'nadam':
        optimizer = tf.keras.optimizers.experimental.Nadam(lr_scheduler)
    elif name_optimizer == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(lr_scheduler)
    elif name_optimizer == 'ftrl':
        optimizer = tf.keras.optimizers.experimental.Ftrl(lr_scheduler)
    elif name_optimizer == 'adamax':
        optimizer = tf.keras.optimizers.experimental.Adamax(lr_scheduler)
    elif name_optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.experimental.Adagrad(lr_scheduler)
    elif name_optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.experimental.Adadelta(lr_scheduler)
    else: 
        print('Name of optimizer not valid!')
        optimizer=None
    return optimizer




''' CALLBACKS '''

class CustomCallback(keras.callbacks.Callback):

    def __init__(self,model_dir):
        super().__init__()
        self.model_dir = model_dir

    def on_epoch_end(self, epoch,logs=None):
        keys = list(logs.keys())
        with open(self.model_dir + 'info.txt', 'a') as f:
            f.write("\nEpoch {}\n".format(epoch))
            for key in keys:
                f.write(' '+key+': '+str(logs[key]))


def choose_callbacks(params,write_dir,model_dir,logdir,lr_scheduler):
   callbacks = []
   # -- CALLBACKS
   early_stopping = keras.callbacks.EarlyStopping(
      monitor="val_loss",
      min_delta=0.0001,
      patience=30,
      verbose=1)
   # tb = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1,write_graph=True)
   checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=model_dir+'checkpoints\\model.{epoch:02d}-{val_loss:.3f}.h5',
                                                monitor = 'val_loss',
                                                save_best_only = True,
                                                save_weights_only = True,
                                                save_freq='epoch')  
   if params['early_stopping']:
      callbacks = [early_stopping]

   if params['lr_scheduler']=='plateau':
      callbacks += [lr_scheduler]
   if write_dir: 
      callbacks += [checkpoint,CustomCallback(model_dir)]
   return callbacks