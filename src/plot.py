import asyncio
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

def plot_cv(history, name, action):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_figwidth(50)
    fig.set_figheight(20)
    fig.suptitle(name, size=70)

    ax1.plot(epochs, acc, 'y', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax1.set_title('Training And Validation Accuracy', size=50)
    ax1.tick_params(labelsize=40)
    ax1.legend(prop={'size': 40})

    ax2.plot(epochs, loss, 'y', label='Training Loss')
    ax2.plot(epochs, val_loss, 'b', label='Validatoin Loss')
    ax2.set_title('Training And Validation Loss', size=50)
    ax2.tick_params(labelsize=40)
    ax2.legend(prop={'size': 40})

    if action is 'save':
        plt.savefig('./diagrams/plots/{name}.png'.format(name = name), format='png')
    elif action is 'show':
        plt.show()

class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.plotting(epoch, logs))
        loop.run_until_complete(task)

    async def plotting(self, epoch, logs={}):
        plt.close('all')

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16, 9))

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.pause(0.0001)
        plt.show(block=False)