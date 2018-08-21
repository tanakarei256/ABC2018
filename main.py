import chainer
import copy
import gzip
import multiprocessing
import numpy
import os
import pickle

import model

with gzip.open('dataset_augmented', mode='rb') as f:
    data = pickle.load(f)

dataset = list(zip(data['train'], data['train_labels']))

def experiment(hyperparameter):
    train_accuracies = []
    test_accuracies  = []
    
    train, test = chainer.datasets.split_dataset_random(dataset, 3000)
    train_x, train_t = chainer.dataset.concat_examples(train)
    test_x, test_t = chainer.dataset.concat_examples(test)

    network = model.Network(hyperparameter)
    optimizer = chainer.optimizers.Adam().setup(network)
    
    best_accuracy = 0

    for max_epoch, batch_size in hyperparameter['iteration']:
        iterator = chainer.iterators.SerialIterator(train, batch_size)        
        while iterator.epoch < max_epoch:
            batch = iterator.next()
            batch_x, batch_t = chainer.dataset.concat_examples(batch)
            
            batch_y = network(batch_x)
            loss = chainer.functions.sigmoid_cross_entropy(batch_y, batch_t)
            network.cleargrads()
            loss.backward()
            optimizer.update()
            
            if iterator.is_new_epoch:
                with chainer.using_config('train', False):
                    with chainer.no_backprop_mode():
                        train_y = network(train_x)
                        train_accuracy = chainer.functions.binary_accuracy(train_y, train_t)

                        test_y = network(test_x)
                        test_accuracy = chainer.functions.binary_accuracy(test_y, test_t)
                
                text = 'Epoch :' + format(iterator.epoch, '4d') + ', TrainAccuracy : ' + format(train_accuracy.data, '1.4f') + ', TestAccuracy : ' + format(test_accuracy.data, '1.4f')
                print(text)
                
                train_accuracies.append(train_accuracy.data)
                test_accuracies.append(test_accuracy.data)
                
                if best_accuracy < test_accuracy.data:
                    best_accuracy = test_accuracy.data
                    best_network = copy.deepcopy(network)
            
    result = {}
    result['train_accuracies'] = train_accuracies
    result['test_accuracies']  = test_accuracies
    result['network']  = best_network
    result['accuracy'] = best_accuracy
    
    return result
        
if __name__=='__main__':
    for i in range(32):
        hyperparameter = {
            'iteration'  : [
                (30, 200),
                (20, 1500)
            ],
            'optimizer'  : chainer.optimizers.Adam,
            'activation_function' : chainer.functions.relu
        }
        result = experiment(hyperparameter)
        
        os.makedirs('result', exist_ok=True)
        with gzip.open('result/' + str(i), mode='wb') as f:
            pickle.dump(result, f)
