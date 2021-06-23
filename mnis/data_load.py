import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


path = Path('/data/mnis/trainingSet/')


# print(df.head())
rows = []
for x in path.iterdir():
    # print(x)
    # print(x.name)
    # print(x.parts)
    # print(x.parts[-1])
    for file in Path(x).iterdir():
        rows.append([x,file.name,x.name])

df = pd.DataFrame(rows,columns=('path',
                           'fineName',
                           'label'))

print(df.head())
print(df.tail())
df=shuffle(df)

print(df.head())
print(df.tail())

