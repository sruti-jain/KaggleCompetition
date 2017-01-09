import pandas
from pandas import DataFrame
from random import shuffle
import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from skimage import exposure
from lasagne import layers
from nolearn.lasagne import NeuralNet

kpTemp = pandas.read_csv("training.csv")
col = kpTemp.columns[:-1].values


def load(test=False, cols=None):

    if test == False:
        kp = pandas.read_csv("training.csv")
    else:
        kp = pandas.read_csv("test.csv")


    kp = kp.fillna(kp.median())
    kp['Image'] = kp['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    print(kp['Image'])

    X = np.vstack(kp['Image'].values)

    p2 = np.percentile(X, 5)
    p98 = np.percentile(X, 95)
    X = exposure.rescale_intensity(X, in_range=(p2, p98))

    X = X / 255
    X = X.astype(np.float32)

    print(kp.columns.values)  # outputs column headings

    if not test:  # raining data
        y = kp[kp.columns[:-1]].values

        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)

    else:
        y = None
    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y


net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
    ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=100,
    verbose=1,
)

X, _ = load(test=True)
X, y = load2d()  # load 2-d data
net2.fit(X, y)
print("This is the value: ")
print(X)
y_pred = net2.predict(X)
y_pred = y_pred * 48 + 48
y_pred = y_pred.clip(0, 96)

df = DataFrame(y_pred, columns=col)

print("Dataframe", df)

lookup_table = pandas.read_csv("IdLookupTable.csv")
values = []

for index, row in lookup_table.iterrows():
    values.append((
        row['RowId'],
        df.ix[row.ImageId - 1][row.FeatureName],
    ))

now_str = datetime.now().isoformat().replace(':', '-')
submission = DataFrame(values, columns=('RowId', 'Location'))
filename = 'submission-{}.csv'.format(now_str)
submission.to_csv(filename, index=False)
print("Wrote {}".format(filename))