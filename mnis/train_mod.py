from mnis.data_load import load_pickle
from mnis.MyModels import generator, discriminator
import numpy as np
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
# from keras.datasets import mnist

path_pic = '/data/mnis/pic_array.pickle'
path_labels = '/data/mnis/pic_labels.pickle'

# df = pic_to_df()
# pic_array, pic_labels = to_np_array(df)
# print(pic_array.shape,pic_labels.shape)
# to_pickle(pic_array,pic_labels,'/data/mnis/pic_array.pickle','/data/mnis/pic_labels.pickle')

X_train, y_labels= load_pickle(path_pic,path_labels)

X_train = (X_train.astype(np.float32)-127.5)/127.5 #Normalize -1 to 1
# print(X_train.shape)
X_train = np.expand_dims(X_train, axis=3)
# print(X_train.shape)
epochs = 40000
batch_size = 128
save_interval = 500
half_batch = int(batch_size / 2)

# print(idx)
# print(len(idx),0,X_train.shape[0])
# print(len(imgs))
# print(noise.shape)

optimizer = Adam(0.0002, 0.5)

generator1 = generator()
generator1.compile(loss='binary_crossentropy',optimizer=optimizer)

discriminator1 = discriminator()
discriminator1.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

z = Input(shape=(100,))
img = generator1(z) ### ???

discriminator1.trainable = False

valid = discriminator1(img)
combined = Model(z,valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator1.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("/data/tmp/mnist_%d.png" % epoch)
    plt.close()


for epoch in range(epochs):
    print(epoch)

    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))

    gen_images = generator1.predict(noise)

    d_loss_real = discriminator1.train_on_batch(imgs,np.ones((half_batch,1)))
    d_loss_fake = discriminator1.train_on_batch(gen_images,np.zeros((half_batch,1)))

    d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

    valid_y = np.array([1] * half_batch)
    g_loss = combined.train_on_batch(noise, valid_y)
    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    if epoch % save_interval == 0:
        save_imgs(epoch)

