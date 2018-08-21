import datetime
import gzip
import numpy
import pickle
import time

def timestr2num(s):
    t = time.strptime(str(s), "b'%H:%M:%S'")
    return t.tm_hour*60*60 + t.tm_min*60 + t.tm_sec

data = dict()
data['train_labels'] = numpy.loadtxt('abc2018dataset/train_labels.csv', dtype=numpy.int32)

data['train'] = []
for i in range(631):
    file_name = 'abc2018dataset/train/' + format(i, '03d') + '.csv'
    data['train'].append(numpy.loadtxt(file_name, delimiter=',', converters = {6: timestr2num}, dtype=numpy.float32))
    
data['test'] = []
for i in range(275):
    file_name = 'abc2018dataset/test/' + format(i, '03d') + '.csv'
    data['test'].append(numpy.loadtxt(file_name, delimiter=',', converters = {6: timestr2num}, dtype=numpy.float32))

with gzip.open('dataset', mode='wb') as f:
    pickle.dump(data, f)
