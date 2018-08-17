import chainer
import gzip
import numpy
import pickle
import os

import model

with gzip.open('../dataset_augmented', mode='rb') as f:
    data = pickle.load(f)

results = []

for i in range(10000):
    file_name = 'result/' + str(i)
    if os.path.exists(file_name):
        with gzip.open(file_name, mode='rb') as f:
            result = pickle.load(f)
        results.append(result)
    else:
        break
        
print(sum(p.data.size for p in results[0]['network'].params()))
        
sorted_results = sorted(results, key=lambda result: -result['accuracy'])

for result in sorted_results:
    print(result['accuracy'])
    
networks = [result['network'] for result in sorted_results][:9]

with chainer.using_config('train', False):
    with chainer.no_backprop_mode():
        submits = []
        for test_data in data['test']:
            test_data = numpy.array(numpy.array(test_data).tolist()).astype(numpy.float32)
            outputs = [network(test_data).data for network in networks]
            output = numpy.median(outputs)
            submits.append(0 if output < 0 else 1)
    
numpy.savetxt('y_submission.txt', numpy.array(submits), delimiter='\n', fmt='%1i')