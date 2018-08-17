import chainer

class Network(chainer.Chain):
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter
        super(Network, self).__init__()
        with self.init_scope():
            self.conv1  = chainer.links.ConvolutionND(1,  4, 50, ksize=4, stride=4)
            self.conv2  = chainer.links.ConvolutionND(1, 50, 50, ksize=4, stride=4)
            self.conv3  = chainer.links.ConvolutionND(1, 50, 50, ksize=1, stride=1)
            self.conv4  = chainer.links.ConvolutionND(1, 50,  1, ksize=1, stride=1)
            self.bnorm1 = chainer.links.BatchNormalization(50)
            self.bnorm2 = chainer.links.BatchNormalization(50)
            self.bnorm3 = chainer.links.BatchNormalization(50)
        
    def __call__(self, h):
        activation_function = self.hyperparameter['activation_function']
        
        h = self.conv1(h)
        h = activation_function(h)
        h = self.bnorm1(h)
        
        h = chainer.functions.max_pooling_nd(h, 4)
        
        h = self.conv2(h)
        h = activation_function(h)
        h = self.bnorm2(h)

        h = self.conv3(h)
        h = activation_function(h)
        h = self.bnorm3(h)
        
        h = self.conv4(h)
        
        h = chainer.functions.average(h, axis=2)

        return h.reshape(-1)