import datetime
import gzip
import numpy
import pickle
import time

def timestr2num(s):
    t = time.strptime(str(s), "b'%H:%M:%S'")
    return t.tm_hour*60*60 + t.tm_min*60 + t.tm_sec
    
def preprocess(trajectory):
    ts = numpy.linspace(0, 29518, 512)
    interpolation = [numpy.interp(ts, trajectory[:, 5], trajectory[:, i]) for i in range(4)]
    return numpy.stack(interpolation, axis=0).astype(numpy.float32)
    
data = dict()
data['train_labels'] = numpy.loadtxt('../abc2018dataset/train_labels.csv', dtype=numpy.int32)

data['train'] = []
for i in range(631):
    file_name = '../abc2018dataset/train/' + format(i, '03d') + '.csv'
    trajectory = numpy.loadtxt(file_name, delimiter=',', converters = {6: timestr2num})
    data['train'].append(preprocess(trajectory))
    
data['test'] = []
for i in range(275):
    file_name = '../abc2018dataset/test/' + format(i, '03d') + '.csv'
    trajectory = numpy.loadtxt(file_name, delimiter=',', converters = {6: timestr2num})
    data['test'].append(preprocess(trajectory))
    
train_array = numpy.array(data['train'])
train_means = numpy.mean(train_array, axis=(0, 2)).reshape(4, -1)
train_stds = numpy.std(train_array, axis=(0, 2)).reshape(4, -1)

data['train'] = [(x - train_means) / train_stds for x in data['train']]
data['test']  = [(x - train_means) / train_stds for x in data['test']]

with gzip.open('dataset', mode='wb') as f:
    pickle.dump(data, f)
