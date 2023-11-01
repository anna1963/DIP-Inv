import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle

def train(attributes):
	if attributes['structure'] == 'kernel_3':
		from models.kernel_3 import NNDC
	elif attributes['structure'] == 'kernel_3_layer_1':
		from models.kernel_3_layer_1_24924 import NNDC
	elif attributes['structure'] == 'kernel_3_layer_5':
		from models.kernel_3_layer_5_25460 import NNDC
	elif attributes['structure'] == 'nearest':
		from models.nearest import NNDC
	elif attributes['structure'] == 'convtranspose':
		from models.convtranspose import NNDC
	else:
		Error = "This combination is not included."
		return Error
	model = NNDC()
	def init_weights(m):
    		if type(m) == nn.Conv2d:
        		torch.nn.init.xavier_uniform_(m.weight, gain = nn.init.calculate_gain('leaky_relu', 0.2))
        		m.bias.data.fill_(0)
	model.apply(init_weights)

	epochs = 100000
        lr_ = 1e-4
        optimizer = optim.Adam(model.parameters(), lr=lr_)
        def theta_m(output):
            theta_m = torch.abs(torch.sub(output.flatten(), torch.ones(output.flatten().size(dim=0)), alpha = -4.605170185988091))
            return torch.sum(theta_m)
        small= 1e20
        epoch = int(0)
        while epoch< epochs and small > 1e-6:
                output = model(input_z)
                m = output.detach().numpy()[0][0].flatten()
                small = min(np.abs(np.max(m)+4.6), np.abs(np.min(m)+4.6))
                loss = theta_m(output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch += 1
                epoch = int(epoch)

        benchmark_epoch = epoch*1
        torch.manual_seed(0)
        np.random.seed(0)

        pkl_name = attributes['file_name'] +'.pkl'
        f = open(pkl_name, 'wb')
        pickle.dump(m, f)
        pickle.dump(benchmark_epoch,f)
        f.close()

        torch.save(model.state_dict(), './start_model/' + attributes['file_name'] +'.pt')

if __name__ == '__main__':
        from time import time

        torch.manual_seed(0)
        np.random.seed(0)
        input_z = torch.normal(0, 10, size=(1, 8))
        attributes = {'structure':'kernel_3',
                      'file_name':'start_weights'
                      }
        start_time = time()
        train(attributes)
        end_time = time()
        print(end_time - start_time)

