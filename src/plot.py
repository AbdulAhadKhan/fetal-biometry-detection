import matplotlib.pyplot as plt

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

    ax1.plot(epochs, acc, 'bo', label='Training Accuracy')
    ax1.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    ax1.set_title('Training And Validation Accuracy', size=50)
    ax1.tick_params(labelsize=40)
    ax1.legend(prop={'size': 40})

    ax2.plot(epochs, loss, 'bo', label='Training Loss')
    ax2.plot(epochs, val_loss, 'b', label='Validatoin Loss')
    ax2.set_title('Training And Validation Loss', size=50)
    ax2.tick_params(labelsize=40)
    ax2.legend(prop={'size': 40})

    if action is 'save':
        plt.savefig('{name}.png'.format(name = name), format='png')
    elif action is 'show':
        plt.show()