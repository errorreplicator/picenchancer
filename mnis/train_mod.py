from mnis.data_load import load_pickle
# import pickle
import numpy as np
# from keras.datasets import mnist

path_pic = '/data/mnis/pic_array.pickle'
path_labels = '/data/mnis/pic_labels.pickle'

# df = pic_to_df()
# pic_array, pic_labels = to_np_array(df)
# print(pic_array.shape,pic_labels.shape)
# to_pickle(pic_array,pic_labels,'/data/mnis/pic_array.pickle','/data/mnis/pic_labels.pickle')

X_train, y_labels= load_pickle(path_pic,path_labels)

X_train = (X_train.astype(np.float32)-127.5)/127.5 #Normalize -1 to 1
print(X_train.shape)
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)