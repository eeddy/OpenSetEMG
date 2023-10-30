import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from os import walk
from libemg.utils import * 

#------------------------------------------------#
#             Make it repeatable                 #
#------------------------------------------------#
def fix_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(gestures, classes, batch_size=5):
    obj = DL_input_data(gestures, classes)
    dl = DataLoader(obj,
    batch_size=batch_size,
    # shuffle=True,
    collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    gestures, labels = [], []
    for gesture, label in batch:
        # concat signals onto list signals
        gestures += [gesture]
        labels += [label]
    # convert back to tensors
    gestures = torch.stack(gestures)
    labels = torch.stack(labels).long()
    return gestures, labels

def extract_adl_data(subject, window_size, window_increment):
    folder = 'Data/S' + str(subject) + '/'
    filenames = next(walk(folder), (None, None, []))[2]  
    adl_files = []
    # Gather all adl files
    for f in filenames:
        if 'sphereo' in f:
            adl_files.append(folder + f)
    # Load all adl files into numpy array

    adl_data = []
    for a in np.sort(adl_files):
        file_data = np.loadtxt(a, delimiter=',')
        if len(adl_data) == 0:
            adl_data = file_data
        adl_data = np.vstack([adl_data, file_data])

    # Read in the prompter file
    p_file = open(folder + 'prompter.txt', 'r')
    p_lines = p_file.readlines()

    adl_start = {'PHONE': {}, 'SPHERO': {}}
    adl_times = {'PHONE': {'walking': [], 'driving': [], 'writing': [], 'typing': [], 'phone': []}, 'SPHERO': {'walking': [], 'driving': [], 'writing': [], 'typing': [], 'phone': []}}

    task = None
    activity = None
    for l in p_lines:
        if 'TASK' in l:
            task = 'PHONE'
            if 'SPHERO' in l:
                task = 'SPHERO'

        split = l.split(',')
        if len(split) == 3:
            activity = split[1]
            adl_start[task][split[1]] = split[0]
        if len(split) == 4 or len(split) == 2:
            adl_times[task][activity].append(split[0])

    labels = ['walking', 'driving', 'writing', 'typing', 'phone']
    data = []
    d_labels = []
    for v, adl in enumerate(labels):
        # Find closest start time
        for task in ['PHONE', 'SPHERO']:
            # Skip for the few people where this didn't work
            if task=='PHONE':
                continue

            s_index = 0
            while(float(adl_data[s_index][0]) < float(adl_start[task][adl])):
                s_index += 1
            e_index = s_index
            while(float(adl_data[e_index][0]) < float(adl_times[task][adl][-1])):
                e_index += 1
            a_d = np.array(adl_data[s_index:e_index, :])

            # Now go through and remove all labels within bounds
            rm_idxs = []
            for i in range(0, 8, 2):
                start = float(adl_times[task][adl][i])
                end = float(adl_times[task][adl][i+1]) # giving extra time to start ADL
                rm_idxs = rm_idxs + list(np.where(((a_d[:,0] > start) & (a_d[:,0] < end)))[0])
            
            # Extract windows from update data
            windows = get_windows(np.delete(a_d, rm_idxs, 0)[:,1:9], window_size, window_increment)
            if len(data) == 0:
                data = windows
            else:
                data = np.vstack([data, windows])
            d_labels = d_labels + [v] * len(windows)

    return data, labels