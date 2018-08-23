## Method
I used 1-dimensional convolutional neural networks for predicting gender of birds.  
The network is defined in model.py.

Trajectories was interpolated to fill small gaps and split into same length avoid long time gaps.
I used longitude, latitude, sun azimuth and sun elevation in trajectories.

## Requirement
* chainer 4.1.0
* numpy 1.14.5

## Usage
1. Place the dataset directory (abc2018dataset) in the same directory as the source code files.
2. Run dataset_pickle.py.
3. Run dataset.py
4. Run main.py
5. Run generate_submission.py
