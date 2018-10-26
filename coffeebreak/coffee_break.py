import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from coffeebreak.callbacks import DecisionBoundaries


def xor_net():
    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_shape=(2, )))
    model.add(Dense(1, activation='sigmoid'))

    return model


xor = xor_net()
xor.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])

x_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y_train = np.array([[0], [1], [1], [0]])

db = DecisionBoundaries(x_train, y_train)
xor.fit(x_train, y_train, epochs=1000, callbacks=[db])
print(xor.predict(np.array([[1, 0], [0, 0], [0, 1], [1, 1]])))
