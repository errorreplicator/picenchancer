

def generator():
    from keras.models import Sequential, Model
    from keras.layers import Dense, BatchNormalization, Reshape, Input
    from keras.layers.advanced_activations import LeakyReLU
    import numpy as np

    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)

    noise_shape = (100,)

    model = Sequential()
    model.add(Dense(256,input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    #According to Image generator flow should be Flatten before tanh (to be checked)
    model.add(Dense(np.product(img_shape), activation='tanh')) # np.product(img_shape) = 28*28*1=784
    model.add(Reshape(img_shape))

    print(model.summary())

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise,img)

def discriminator():
    from keras.models import Sequential, Model
    from keras.layers import Dense, BatchNormalization, Reshape, Input, Flatten
    from keras.layers.advanced_activations import LeakyReLU
    import numpy as np

    img_rows = 28
    img_cols = 28
    channels = 1
    img_shape = (img_rows, img_cols, channels)


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation='sigmoid'))

    print(model.summary())

    img = (Input(shape=img_shape))
    validity = model(img)
    return Model(img,validity)

generator()