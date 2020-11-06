import keras
from keras.models import Sequential
#from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class TradeNetwork:
    def __init__(self):
        self.neural_net = Sequential()
        
    def create_neural_network(self, num_input_vars, num_target_vars, classifier = True):
        if not isinstance(num_input_vars, int):
            if not num_input_vars.is_integer():
                raise ValueError('{} is not an integer'.format(num_input_vars))
        if not isinstance(num_target_vars, int):
            if not num_target_vars.is_integer():
                raise ValueError('{} is not an integer'.format(num_target_vars))
        if num_input_vars<num_target_vars:
            raise ValueError('{}'.format(num_input_vars, num_target_vars))
        neurons = num_input_vars
        while neurons > num_target_vars:    
            self.neural_net.add(Dense(neurons, activation = 'relu'))
            neurons = neurons/2
        if classifier:
            self.neural_net.add(Dense(num_target_vars, activation = 'softmax'))
        else:
            self.neural_net.add(Dense(num_target_vars, activation = 'relu'))
