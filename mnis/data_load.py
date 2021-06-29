import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


path = Path('/data/mnis/trainingSet/')


# print(df.head())
def pic_to_df(path_dir = path):
    rows = []
    for x in path_dir.iterdir():
        # print(x)
        # print(x.name)
        # print(x.parts)
        # print(x.parts[-1])
        for file in Path(x).iterdir():
            rows.append([x,file.name,x.name])

    df = pd.DataFrame(rows,columns=('path',
                               'fileName',
                               'label'))

    # print(df.head())
    # print(df.tail())
    df=shuffle(df)
    return df
    # print(df.head())
    # print(df.tail())

def to_np_array(df):
    from PIL import Image
    import numpy as np
    images = []
    y_label = []
    idx = 0
    for index, x in df.iterrows():
        fullPath = str(x['path']) + '/' + str(x['fileName'])
        # print(fullPath)
        image = Image.open(fullPath)
        # print(image.format)
        # print(image.size)
        # print(image.mode)
        # print(x)
        images.append(np.asarray(image))
        y_label.append(int(x['label']))
        idx+=1
        if idx % 1000 == 0:
            print(idx)
    return np.asarray(images), np.asarray(y_label).reshape(-1,1)

def to_pickle(pic_array, pic_labels,filePath1,filePath2):
    import pickle
    from pathlib import Path
    my_path = Path(filePath1)
    with my_path.open('wb') as fp:
        pickle.dump(pic_array, fp)

    my_path = Path(filePath2)
    with my_path.open('wb') as fp:
        pickle.dump(pic_labels, fp)

def load_pickle(filePath1,filePath2):
    import pickle
    pic_array = pickle.load(open(filePath1, 'rb'))
    pic_labels = pickle.load(open(filePath2, 'rb'))
    return pic_array,pic_labels

