from torch.utils.data import Dataset
import h5py
import util
import numpy as np 
from tqdm.autonotebook import tqdm
import os
from glob import glob
import pickle
import torch

class QuenchDataset(Dataset):
    '''
    positive_files = glob(down_folder + '*.h5')
    negative_files = glob(up_folder + '*.h5')

    print('balacing the dataset, repeating the negative files 4 times')
    negative_files = negative_files

    print(f'>pos {len(positive_files)} neg {len(negative_files)}')

    dataset_ = dataset.QuenchDataset(positive_files, negative_files, input_dims, output_dims)
    '''
    def __init__(self, positive_files,negative_files, input_dim, output_dim, num_sampled_locations = 1, signal_subsampling = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.positive_files = positive_files
        self.negative_files = negative_files
        self.files = self.positive_files + self.negative_files
        self.num_sampled_locations = num_sampled_locations
        self.signal_subsampling = signal_subsampling
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files)
        
        self.valid_locations = sorted(list(set([location 
             for file in tqdm(self.files)
             for location in util.read_h5_llrf_locations(file) 
             if location.startswith('C')])))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx, locations = None):
        data = self.process_file(self.files[idx], locations)
        data['label'] = len(data['Y']) * [self.labels[idx]]
        return data
    
    def process_file(self, file, locations = None, threshold = 1):
        if locations is None:
            locations = self.sample_locations(file, self.num_sampled_locations)
        
        data = self.read_data(file, locations)
        return {'X' : [], # self.prepare_data(data,self.input_dim, False), \
                'Y' : self.prepare_data(data,self.output_dim, threshold)}
    
    def read_data(self,file, locations = None):
        with h5py.File(file,'r') as f:
            return [np.array(f['LLRF'][location]['data']) for location in locations]
        
    def read_locations(self, file):
        with h5py.File(file,'r') as f:
            locations = [k for k in f['LLRF'].keys() if k in self.valid_locations]
            assert(len(locations) > 0)
            return locations

    def sample_locations(self, file, num_sampled_locations):
        ''' this function sampels @num_sampled_locations which are in the file and are in valid_location at the same time '''
        locations = self.read_locations(file)
        
        return [locations[i] for i in np.random.randint(0,len(locations),num_sampled_locations)]
        
    def prepare_data(self, data, dims, threshold):
        prepared_data = []
        for datum in data:
            # [6,1820,250] - [dimensions, time, num of signals]
            datum = datum[dims,...]
            keep = datum[0,:,:].mean(0) > threshold
            if keep.sum() != 0:
                datum = datum[:,:,keep]
            
            for dim in range(len(datum)):
                datum[dim,...] = (datum[dim,...] - datum[dim,...].min()) / (datum[dim,...].max() - datum[dim,...].min())
                datum[dim,...] = datum[dim,...] * 2 -1
            
            # subsampling
            datum = datum[:,::self.signal_subsampling,:]
            
            prepared_data.append(datum)
        return prepared_data

def concatenate_data(data): 
    cdata = {'X' : [], 'Y' : [], 'label' : []}
    for datum in data:
        for key in cdata.keys():
            cdata[key] = cdata[key] + datum[key]
    return cdata

def h5py_files_to_one_numpy(folder, dimensions, subsampling = 10, T = 1000):
    '''
    Function reads files from @folder, filters out all signals where at least one of amplitudes 
    (dimensions f['LLRF']{[0,...], [2,...], [4,...]}) is below @T, selects @dimensions and subsample 
    them with rate @subsampling, i.e. f['LLRF'][@dims,::@subsampling,:]
    
    @folder folder with h5 files
    @dimensions stored dimensions
    @subsampling subsampling rate of signals (usually ~1820)
    @T threshold of the input implidutes
    @verbose
    
    @returns a list of signals    
    '''
    files = sorted(glob(folder + '*.h5'))
    print(files)
    
    files_and_locations = []
    pids = []
    X = []
    
    for file in tqdm(files):
        with h5py.File(file,'r') as f:
            for key in f['LLRF']:
                y = np.array(f['LLRF'][key]['data']).astype(np.float32)
                # sequences to keep (they are empty if < 1k)
                keep = np.bitwise_and(np.bitwise_and(y[0,:,:].sum(0) > T,y[2,:,:].sum(0) > T),y[4,:,:].sum(0) > T)
                y = y[dimensions,::subsampling,:]
                
                # pid data are in file, stored as an extra dimension
                if 'EVENT_INFO/pid' in f:
                    pid = np.array(f['EVENT_INFO/pid']).astype(np.uint64)
                    pid = pid[keep]
                    pids.append(pid)
                else:
                    pids.append([])

                y = y[...,keep]
                
                X.append(y)
                files_and_locations.append([file, key])
    return X,files_and_locations, pids

def load_data_from_numpy(folder, file, dimensions):
    if not os.path.isfile(file):
        print(f'file {file} does not exist, loading all h5 files from {folder}')
        X, files_and_locations, pids = h5py_files_to_one_numpy(folder, dimensions)
        print(f'data loaded, saving to {file}')
        with open(file,'wb') as f:
            pickle.dump([X, files_and_locations, pids],f)
    else:
        print(f'file {file} exists, loading file')
        with open(file,'rb') as f:
            X, files_and_locations, pids = pickle.load(f)
        print(f'file {file} loaded')
    
    return X, files_and_locations, pids

##
## NORMALIZATION
##

def normalize_data(X, normalize_dimensions_independently = True):
    '''
    Normalizing list @X of quench data to range (-1,1)
    a format of each X[...] must be [dims, signals, events] (e.g. X[i].shape := (3, 182, 207))
    '''
    for i in tqdm(range(len(X))):
        x = X[i]
        n_dims = x.shape[0]
        n_vals = x.shape[1]
        
        # all values must be positive
        min = np.where(np.amin(x,1) < 0)
        x[min[0],:,min[1]] = x[min[0],:,min[1]] - np.amin(x[min[0],:,min[1]],1).reshape((-1,1))
        
        
        '''
        # normalisation of all pulses jointly (accorindg to maxes of all pulses)
        
        
        # often - full [6,1820,~250] reduced [3,182,~250]
        n_dims, n_vals, n_pulses = x.shape

        if normalize_dimensions_independently:
            for dim in range(n_dims):
                x[dim,...] = x[dim,...] / x[dim,...].max()
        else:
            x = x / x.max()

        x = x.transpose((2,0,1)).reshape((n_pulses,1,n_dims * n_vals))
        x = (x - 0.5) * 2
        '''
        # normalising each pulse individually (i.e. max for each pulse)
    
        # getting it into range (-1,1)
        x = x / np.amax(x,1).reshape((n_dims,1,-1))
        x = 2 * (x - 0.5)

        x = x.reshape((n_dims * n_vals,-1))
        x = x.T
        # x = x[np.isnan(x).sum(1)==0,:]
        x = x.reshape((-1,1,n_dims * n_vals))
        
        X[i] = torch.tensor(x)
    return X

def forward_differentiate_data(X):
    Xd = []
    for i in tqdm(range(len(X))):
        x = X[i]
        x = x.reshape((-1,3,182))
        x = x[:,:,-1:] - x[:,:,:-1]
        x = x.reshape((-1,1,3 * 181))
        Xd += [x]
    return Xd

def generate_permutation_for_batch_indices(input, n_batches, seed = 0):
    '''
    function generates list of lists, where each sublists contains a permuation of range(len(@input)) indices split into n_batches
    '''
    N = len(input) 
    n_samples_per_batch = int( (N - N % n_batches) / n_batches)

    np.random.seed(seed)
    perm = np.random.permutation(len(input))
    perm = perm[:n_batches * n_samples_per_batch].reshape((n_batches,n_samples_per_batch)).tolist() + [perm[n_batches * n_samples_per_batch:].tolist()]
    return perm


##
## PADDING
##
def pad_sequence_inputs(input, PAD = -2):
    for i in range(len(input)):
        x = input[i]
        pad = PAD * torch.ones((1,x.shape[1],x.shape[2]))
        input[i] = np.concatenate((x,pad),0)
    return input

def pad_inputs_to_uniform_length(input, seq_length, PAD = -2):
    # seq_length = np.max([input[i].shape[0] for i in range(len(input))]) + 1 
    for i in range(len(input)):
        x = input[i]
        pad = PAD * np.ones( (seq_length - x.shape[0],x.shape[1],x.shape[2]))
        input[i] = np.concatenate((x,pad),0)
    return input

# --- IQ transformation from phase + ampl. --- 

def ampl_phase_to_IQ(A, P):
    '''
    tranfroms phase @P and amplitude @A to real and imaginary components. 
    @A - amplitude
    @P - phase (in degs)
    '''
    re = A * np.cos(np.deg2rad(P))
    im = A * np.sin(np.deg2rad(P))
    return re, im

def transform_RF_pulses(X, func = ampl_phase_to_IQ):
    '''
    Transforms a list of inputs according to function @func
    '''
    for i in tqdm(range(len(X))):
        X[i][0,...],X[i][1,...] = func(X[i][0,...],X[i][1,...])
        X[i][2,...],X[i][3,...] = func(X[i][2,...],X[i][3,...])
        X[i][4,...],X[i][5,...] = func(X[i][4,...],X[i][5,...])
    return X

def select_dimensions(input, input_dims):
    return [x[input_dims,...] for x in input]

# --- SCALING ---
def scale_raw_input(x,cal_samples = range(9,179)):
    '''
    The function rescales forward (F) and reflected (R) @x s.t. their scale fits to amplitude 
    
    (F;R) * c = A
    
    This is a condition, that must be fulfilled. LSQ is found in complex coordinates.
    
    
    Code based on conversation with Annika Eichler:
    x must be in a format [dims, signals (e.g. 182), pulses (e.g. ~200)]
    MATLAB
    >>> x = h5read(filename,['/LLRF/',tabRuns.location{ind},'/data']);
    >>> cal_samples = 100:1800;
    >>> A = (squeeze(x(1,:,1:2:5))).*exp(1i*(squeeze(x(1,:,2:2:6)))/180*pi);
    >>> coeff = [A(cal_samples,2) A(cal_samples,3)]\A(cal_samples,1);
    >>> B = (squeeze(x(index,:,1:2:5))).*exp(1i*(squeeze(x(index,:,2:2:6)))/180*pi);

    Then the probe is B(:,1), the forward is coeff(1).*B(:,2) und reflected is 
    coeff(2).*B(:,3) as complex signals, from which you can derive either representation that you want.

    '''
    assert x.shape[0] == 6
    assert x.shape[1] % 182 == 0 # multiple of 182 (either 182, or 1820)
    assert x.shape[2] > 0 # at least one pulse
    
    x = x[[0,2,4],...]*np.exp(1j*(np.deg2rad(x[[1,3,5],...]))) 
    coeff = np.linalg.lstsq(x[1:3,cal_samples,0].T,x[0:1,cal_samples,0].T)[0]
    # return np.stack((x[0,...],coeff[0] * x[1,...], coeff[1] * x[2,...]),0)
    return np.stack((np.absolute(x[0,...]), np.angle(x[0,...],deg = True),
                     np.absolute(coeff[0] * x[1,...]), np.angle(coeff[0] * x[1,...], deg = True), 
                     np.absolute(coeff[1] * x[2,...]), np.angle(coeff[1] * x[2,...], deg = True)),0)


def scale_raw_inputs(X, cal_samples = range(9,179)):
    '''
    does the same as @scale_raw_input, but on a list of inputs
    '''
    return [scale_raw_input(x,cal_samples) for x in tqdm(X)]

# --- Helping function for loading list of datasts stored in Numpy pickles ---
def load_multiple_data_from_numpy(files, folders, dims):
    '''
    this function just usses
    '''
    
    X = []
    files_and_locations = []
    pids = []
    
    for i in range(len(files)):
        X_, files_and_locations_, pids_ = load_data_from_numpy(folders[i], files[i], dims)

        X += X_
        files_and_locations += files_and_locations_
        pids += pids_

    return X, files_and_locations, pids

