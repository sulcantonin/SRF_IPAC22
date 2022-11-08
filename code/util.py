from scipy.interpolate import UnivariateSpline
from scipy.cluster.hierarchy import dendrogram

import numpy as np
import h5py
from copy import copy
import torch

forward_diff = lambda f : f[1:] - f[:-1]

def smoothen_X(X,var_mult = 0.01):
    x = range(X.shape[0])
    X_APPROX = np.zeros_like(X)
    for i in range(X.shape[1]):
        y = X[:,i]
        spline = UnivariateSpline(x,y,w = None,k = 2,s = var_mult * np.var(y))
        X_APPROX[:,i] = spline(x)
    
    # plt.subplot(221)
    # plt.plot(X,'b')
    # plt.subplot(221)
    # plt.plot(X_APPROX,'r--')
    
    return X_APPROX

def read_h5_llrf_file(file,location):
    '''
    Getting data from a specific h5py file, the file structure must be 
    f['LLRF'][location][data] which is (6,1820,~250) array of floats
    pids is a set of event identifiers (macropulses?)
    '''
    
    d = None
    pids = None
    
    with h5py.File(file,'r') as f:
        # the data must be copied, they are moved from memory after the file is closed! 
        if 'LLRF' in f.keys():
            if location in f['LLRF'].keys():
                if 'data' in f['LLRF'][location].keys():
                    d = np.array(f['LLRF'][location]['data']).copy()
        if 'EVENT_INFO' in f.keys():
            if 'pid' in f['EVENT_INFO']:
                pids = np.array(f['EVENT_INFO']['pid']).copy()
    return d,pids

def read_h5_llrf_locations(file):
    locations = []
    with h5py.File(file,'r') as f:
        if 'LLRF' in f.keys():
            locations = list(f['LLRF'].keys())
        return locations

def points_in_std_bounds(d,std_mult = 1):
    '''
    @d (....,N) where N is number of observations 
    @returns True forall points whose mean is below std_mult * std(mean(d)), False otherwise
    '''
    
    # last dimension are individual samples, calculating their sum through all dimensions
    d_ = d.reshape((-1,d.shape[-1]))
    
    # mean
    m = d_.mean(0)
    # standard deviation 
    std = np.std(d_,0)
    # filtering all dims which are below std
    return np.abs(m - m.mean()) < std_mult * std

def points_above(d,T = 2.5):
    '''
    @d (....,N) where N is number of observations 
    @returns True forall points whose mean value is above T
    '''
    
    # last dimension are individual samples, calculating their sum through all dimensions
    d_ = d.reshape((-1,d.shape[-1]))
    
    # mean
    m = d_.mean(0)
    # filtering all dims which are below std
    return m > T
def most_likely_input(model,input_shape, label):
    inputs = torch.ones(input_shape,device = device,requires_grad=True)
    
    # L_pos = torch.ones((inputs.shape[0],1,1)).to(device)
    # L_neg = torch.ones((inputs.shape[0],1,1)).to(device)
                
    # freezes weights of model
    for param in model.parameters():
        param.requires_grad = False

    
    # inputs = D[0].to(device).requires_grad_(True)
    optimizer_inputs = torch.optim.Adam(params = [inputs],lr=0.1)

    

    for i in tqdm(range(1000)):
        loss = criterion(label[-1],model(inputs)[0][-1])
        optimizer_inputs.zero_grad()
        loss.backward()
        optimizer_inputs.step()

        if i % 100 == 0:
            print(loss)
            
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def weight(x):
    ''' this is weighting function for loss function for the anomaly detection on sequences
    it measures eucl. distance of all pulses ([pulses,1,signals]) from a mean pulse
    '''
    w = torch.linalg.vector_norm(x[1:,...] - x[:-1,...],axis = (1,2))
    w = torch.cumsum(w,0)
    w = torch.cat((w[-1:] * 0, w),0)
    # w = torch.cat((w,w[-1:]),0)
    w = w / w.sum()
    return w 

    '''
    ll = (torch.where(L == -1)[0])
    for i in ll:
        x = D[i]
        w = weight(x)

        plt.subplot(221)
        plt.plot(t2n(x[:,0,:].T))
        plt.subplot(222)
        plt.plot(t2n(w))
        plt.subplot(223)
        plt.plot(t2n(outputs[i][1] ** (-1)))
        plt.subplot(224)
        plt.plot(t2n(torch.mul(weight(D[i]).ravel(),(outputs[i][1] ** (-1)).ravel())))
        print(w.sum())
        plt.show()
    
    '''