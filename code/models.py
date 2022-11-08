import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuenchDetectorNetwork(nn.Module):
    def __init__(self, eta_anom, eta_label, eta_network):
        super(QuenchDetectorNetwork, self).__init__()

        self.eta_anom = eta_anom
        self.eta_label = eta_label
        self.eta_network = eta_network

    def set_all_weights(self, value = 0.0001):
        with torch.no_grad():
            for name,param in self.named_parameters():
                if 'weight' in name:
                    param[:] = value
                elif 'bias' in name:
                    param[:] = value

    def hidden_F_norm(self):
        norm = []
        for name,param in self.named_parameters():
            if 'weight' in name:
                norm += [torch.linalg.matrix_norm(param).flatten()]
            elif 'bias' in name:
                norm += [torch.linalg.matrix_norm(param.view((1,-1))).flatten()]
        return torch.cat(norm).mean()

    def forward(self, x):
        score = (self.phi(x) - self.c)
        # score.shape = [|seq|, 1, |hidden|]
        # norm = torch.linalg.vector_norm(score,dim=-1)
        # return norm.mean(), norm
        return torch.linalg.vector_norm(score[-1,...],dim=-1), torch.linalg.vector_norm(score,dim=-1, keepdim = True)

    def loss(self,inputs,labels):

        N = len(inputs)
        # calculating outputs of the network
        outputs = [self(x) for x in inputs]

        # unsupervised data term
        if self.eta_anom > 0.0:
            # anomaly loss, check forward function
            loss_R = self.eta_anom   * torch.vstack([outputs[i][1].mean() for i in range(N)]).mean()
        else:
            loss_R = torch.tensor([0.0], device = inputs[0].device)

        # semi-supervised data term
        if self.eta_label > 0.0:
            # data loss, normal events - all pulses are fit, faulty - only the last pulse is considered as faulty
            loss_D = self.eta_label   * torch.vstack([outputs[i][1].mean()
                                                 if   labels[i] == 1
                                                 else outputs[i][1][-1] ** labels[i] # labels[i] == -1
                                                 for i in range(N)]).mean()
        else:
            loss_D = torch.tensor([0.0], device = inputs[0].device)

        # reguralization
        if self.eta_network > 0.0:
            loss_W = self.eta_network * self.hidden_F_norm()
        else:
            loss_W = torch.tensor([0.0], device = inputs[0].device)

        return loss_R, loss_D, loss_W

class QuenchDetectorNetworkClassifier(nn.Module):
    def __init__(self, criterion = nn.BCEWithLogitsLoss()):
        super(QuenchDetectorNetworkClassifier, self).__init__()
        self.criterion = criterion

    def forward(self, x):
        score = self.phi(x)
        # score.shape = [|seq|, 1, |hidden|]
        # norm = torch.linalg.vector_norm(score,dim=-1)
        # return norm.mean(), norm
        return score[-1,...], score

    def loss(self,inputs,labels, device):

        N = len(inputs)

        # calculating outputs of the network
        outputs = [self(x)[1] for x in inputs]
        loss = torch.vstack([self.criterion(outputs[i][:,0,0], torch.tensor([labels[i]] * outputs[i].shape[0]).to(device))
                             if   labels[i] == 1
                             else self.criterion(outputs[i][-1,0,0], 0.0 * torch.tensor(labels[i]))
                             for i in range(N)]).mean() # both_in_D
        return torch.tensor([0], device = device), loss, torch.tensor([0], device = device)

class QuenchDetectionNetworkLSTM(QuenchDetectorNetwork):
    def __init__(self,num_hidden, num_hidden_layers, eta_anom, eta_label, eta_network, num_dims = 6, dropout = 0.0):
        super().__init__(eta_anom, eta_label, eta_network)
        self.num_dims = num_dims
        self.num_inputs = num_dims * 182
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.input_dropout = dropout
        self.LSTM_dropout = dropout

        IPAC_version = False # biases, trainable c
        if IPAC_version:
            self.c = nn.Parameter(torch.zeros(self.num_hidden))
        else:
            self.c = torch.randn(self.num_hidden, requires_grad = False) # fix 20220820, this should be fixed and non-zero

        self.input_rnn  = nn.LSTM(self.num_inputs, self.num_hidden,
                                  num_layers = self.num_hidden_layers,
                                  dropout = self.LSTM_dropout,
                                  bias = IPAC_version)
        self.hidden_rnn = nn.LSTM(self.num_hidden,self.num_hidden,
                                  num_layers = self.num_hidden_layers,
                                  dropout = self.LSTM_dropout,
                                  bias = IPAC_version)
        self.hidden_lin = nn.Linear(self.num_hidden, self.num_hidden,
                                    bias = IPAC_version)
        self.dropout = nn.Dropout(p = self.input_dropout)

    def phi(self,x):
        phi = x
        phi, _ = self.input_rnn(phi)
        phi = self.dropout(phi)
        phi, _ = self.hidden_rnn(phi)
        phi = self.dropout(phi)
        return self.hidden_lin(phi)

class QuenchDetectionNetworkClassifierLSTM(QuenchDetectorNetworkClassifier):
    def __init__(self,num_hidden, num_hidden_layers, eta_anom, eta_label, eta_network, num_dims = 6, dropout = 0.0):
        super().__init__()
        self.num_dims = num_dims
        self.num_inputs = num_dims * 182
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.input_dropout = dropout
        self.LSTM_dropout = dropout

         # self.c = nn.Parameter(torch.zeros(self.num_hidden)) # fix 20220820, this should be fixed and non-zero
        self.c = torch.randn(self.num_hidden, requires_grad = False) # fix 20220820, this should be fixed and non-zero

        self.input_rnn = nn.LSTM(self.num_inputs,self.num_hidden)
        self.hidden_rnn = nn.LSTM(self.num_hidden,self.num_hidden,num_layers = self.num_hidden_layers, dropout = self.LSTM_dropout)
        self.hidden_lin = nn.Linear(self.num_hidden, 1)
        self.dropout = nn.Dropout(p = self.input_dropout)

    def phi(self,x):
        phi = x
        phi, _ = self.input_rnn(phi)
        phi = self.dropout(phi)
        phi, _ = self.hidden_rnn(phi)
        phi = self.dropout(phi)
        return self.hidden_lin(phi)


def show_input_gradients_for_output(x, model, device, num_dims):
    '''
    Inspired by https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709

    show_input_gradients_for_output(input_[0],model)
    '''
    x = torch.autograd.Variable(x,requires_grad=True).to(device)
    grad = torch.autograd.grad(model(x)[1][-1,0,0], x)
    assert len(grad) == 1
    grad = grad[0].reshape((-1,num_dims,182))
    return grad.cpu().detach().numpy() # plt.plot(torch.abs(grad[:,0,:].cpu().detach().numpy()[-1,:]))
