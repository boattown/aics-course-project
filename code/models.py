import torch.nn as nn

class Probe(nn.Module):
    '''
    construct model for binary classification
    '''
    def __init__(self):
        super(Probe, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(1)
        
    def forward(self, obj, affordance):
        '''
        combine object and affordance vectors by multiplication and pass it through 
        (i) Sigmoid function, (ii) linear layer, (iii) LogSoftmax
        result: 0 or 1, indicating if the affordance belongs to the object or not
        '''
        combined_vector = obj * affordance
        x1 = self.sigmoid(combined_vector)
        x2 = self.fc1(x1)
        output = self.softmax(x2)
        return output