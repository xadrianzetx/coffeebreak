import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class DecisionBoundaries(Callback):
    def __init__(self, x, y):
        Callback.__init__(self)
        self._x = x
        self._y = y
        self._epoch = 1

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0 or epoch == 999:
            cmap = plt.get_cmap('Paired')

            xmin, xmax = self._x[:, 0].min(), self._x[:, 0].max()
            ymin, ymax = self._x[:, 1].min(), self._x[:, 1].max()
            steps = 1000
            x_span = np.linspace(xmin, xmax, steps)
            y_span = np.linspace(ymin, ymax, steps)
            xx, yy = np.meshgrid(x_span, y_span)

            labels = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

            z = labels.reshape(xx.shape)
            fig, ax = plt.subplots()
            ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

            colors = ['steelblue' if label == 1 else 'darkred' for label in self._y]
            ax.scatter(self._x[:, 0], self._x[:, 1], c=colors, cmap=cmap, lw=0)
            plt.title('Neurons: 2 Epoch: {}'.format(epoch))
            plt.savefig('./temp/{}.png'.format(epoch))
            plt.close()

            self._epoch += 1
