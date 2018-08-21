import gzip
import numpy
import pickle
import random

trim_duration = 60 * 60 * 4
data_size = 256
minimum_data_size = 200
minimum_interval = 250

def trim_trajectory(trajectory, max_trajectories = 10000):
    trimed_trajectories = []
    prev_i = -100000000

    for i in range(0, trajectory.shape[0], 10):
        if i + minimum_data_size >= trajectory.shape[0]:
            break
            
        if trajectory[i + minimum_data_size, 5] > trajectory[i, 5] + trim_duration:
            continue
        
        if i < prev_i + minimum_interval:
            continue
        
        ts = numpy.linspace(trajectory[i, 5], trajectory[i, 5] + trim_duration, data_size)
        interpolation = [numpy.interp(ts, trajectory[:, 5], trajectory[:, i]) for i in range(4)]
        interpolation =  numpy.stack(interpolation, axis=0).astype(numpy.float32)
        trimed_trajectories.append(interpolation)
        
        prev_i = i
        
        if len(trimed_trajectories) > max_trajectories:
            break
            
    return trimed_trajectories

with gzip.open('dataset', mode='rb') as f:
    data = pickle.load(f)

new_data = {}

new_data['train'] = []
new_data['train_labels'] = []

for trajectory, label in zip(data['train'], data['train_labels']):
    trimed_trajectories = trim_trajectory(trajectory)
    
    if len(trimed_trajectories) > 10:
        trimed_trajectories = random.sample(trimed_trajectories, 10)
    
    new_labels = [label] * len(trimed_trajectories)
    
    new_data['train'].extend(trimed_trajectories)
    new_data['train_labels'].extend(new_labels)
    
new_data['test'] = []

for trajectory in data['test']:
    trimed_trajectories = trim_trajectory(trajectory)
    
    new_data['test'].append(trimed_trajectories)
    
train_array = numpy.array(new_data['train'])
train_means = numpy.mean(train_array, axis=(0, 2)).reshape(4, -1)
train_stds = numpy.std(train_array, axis=(0, 2)).reshape(4, -1)

new_data['train'] = [(x - train_means) / train_stds for x in new_data['train']]
new_data['test']  = [[(y - train_means) / train_stds for y in x] for x in new_data['test']]

new_data['train_labels'] = numpy.array(new_data['train_labels'], dtype=numpy.int32)

print('train data : ' + str(len(new_data['train'])))

with gzip.open('dataset_augmented', mode='wb') as f:
    pickle.dump(new_data, f)