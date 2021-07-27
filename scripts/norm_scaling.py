from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

def normalize(x):
    norm = MinMaxScaler()
    return norm.fit_transform(x)
