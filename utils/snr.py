import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch 

import numpy as np
import random
import scipy.linalg


def batch_data(batch_size=32, width=5):
    x = torch.FloatTensor(batch_size)
    for i in range(batch_size):
        x[i] = np.random.uniform(-width, width)
    y = torch.sin(x)

    return x, y

class MLP(nn.Module):
    """
    Simple 3-layer MLP for learning a sine wave
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 30)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 15)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(15, 1)
        self.prediction = nn.Tanh()

    def forward(self, inputs):
        out1 = self.rl1(self.fc1(inputs))
        out2 = self.rl2(self.fc2(out1))
        preds = self.prediction(self.fc3(out2))
        return preds

class SNR(optim.Optimizer):
    """
    Implements a "Gauss-Newton" optimizer: 
        
        theta_t+1 = theta_t - inv(Jac_g^T Jac_g) * grad_f

    Here, f(theta) is computed via backpropagation, and is the
    gradient of the network loss wrt the parameters (a N dim vector).

    g(theta) = (target - prediction)

    where g(theta) is the vector of the differences between the targets
    and the predictions. Note that the MSE loss is the sum of squares 
    of g(theta) divided by the minibatch size. Jac_g is the Jacobian of g
    wrt the parameters of the network. 
    
    """
    def __init__(self, params):
        defaults = {}
        super(SNR, self).__init__(params, defaults)
        self.A_t = None

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        grads = []    
        step = 0
        # flatten param groups into single Tensor of size [n_params]
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                s = p.grad.size()
                if len(s) == 2:
                    grads.append(p.grad.view(s[0] * s[1]))
                else:
                    grads.append(p.grad)
                state = self.state[p]

            # State init
            if len(state) == 0:
                state['step'] = 0
                
            state['step'] += 1
            step = state['step']

        # TODO: Change to Gauss-Newton computation 

        
        grads = torch.cat(grads)
    
        # initialize A_t matrix
        if self.A_t is None:
            self.A_t = 100 * torch.eye(len(grads)).cuda()

        # outer product
        outer = torch.ger(grads, grads).data

        gamma = (1. / step) ** 0.85
        alpha = 1. / step

        self.A_t += gamma * (outer - self.A_t)
        
        if step < 2000: 
            inv_outer = -1 * torch.eye(len(grads)).cuda()
        else:
            # inverse
            damping = 0.00001 * torch.eye(len(grads)).cuda()
            inv_outer = torch.inverse(damping + self.A_t)
        #inv_outer = torch.FloatTensor(np.linalg.pinv(self.A_t.numpy()))
        # matrix vector product
        delta = alpha * torch.mv(inv_outer, grads.data)

        start = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                s = p.grad.size()

                if len(s) == 2:
                    p_len = s[0] * s[1]
                else:
                    p_len = s[0]
                
                p.data = p.data.add(-delta[start:start+p_len])
                start += p_len

if __name__ == '__main__':
    model = MLP()
    print(model)
    loss_fn = nn.MSELoss()
    optimizer = SNR(model.parameters())
    #optimizer = optim.Adam(model.parameters(), lr = 0.01)
    for i in range(100):
        data = batch_data(32)
        x = Variable(data[0].view(32, 1))
        optimizer.zero_grad()
        p = model(x)
        y = Variable(data[1].view(32, 1))
        l = loss_fn(p, y)
        if i % 10 == 0:
            print(l.data[0])
        l.backward()
        optimizer.step()

    print('<========= BEGIN TEST ==========>')

    # test
    for i in range(500):
        data = batch_data(32, 6)
        x = Variable(data[0].view(32, 1))
        p = model(x)
        y = Variable(data[1].view(32, 1))
        l = loss_fn(p, y)
        if i % 10 == 0:
            print(l.data[0])

    
