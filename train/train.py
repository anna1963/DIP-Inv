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

import SimPEG.data_misfit as data_misfit

from SimPEG.electromagnetics.static import resistivity as dc

from discretize import TreeMesh, TensorMesh

from SimPEG.electromagnetics.static.utils.static_utils import (
generate_dcip_sources_line,
apparent_resistivity_from_voltage,
plot_pseudosection,
)

from SimPEG import (
maps,
data,
data_misfit,
regularization,
optimization,
inverse_problem,
inversion,
directives,
utils,
)

from SimPEG.utils import sdiag

from SimPEG import SolverLU as Solver

from discretize.utils import mkvc, refine_tree_xyz, active_from_xyz

from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import copy
from time import time


def train(attributes):
    """
    attributes = {'structure': architecture of the model
                      'Dropout':True for adding a dropout layer, False for not adding any dropout layer
                      'Case':Name of the inverting case
                      'decay_rate':tau in the beta decay curve
                      'lr':learning rate for Adam
                      'file_name':the output file name
                      'uncertainty':0.05 for both case 1.X and 2.X
                      }
    """
    if attributes['structure'] == 'kernel_3':
        if attributes['dropout_rate'] == 0.1:
            from models.kernel_3 import NNDC_dropout_10
            model = NNDC_dropout_10()
        elif attributes['dropout_rate'] == 0.05:
            from models.kernel_3 import NNDC_dropout_5
            model = NNDC_dropout_5()
        elif attributes['dropout_rate'] == 0.0:
            from models.kernel_3 import NNDC
            model = NNDC()
        else:
            return "Error: The dropout_rate is not valid"
        from models.kernel_3 import NNDC
        model.load_state_dict(torch.load('./start_model/start_weights.pt'))
    elif attributes['structure'] == 'kernel_3_layer_1':
        if attributes['dropout_rate'] != 0.0:
            print("Wrong choice for Dropout")
            Error = "This combination is not included."
            return Error
        else:
            from models.kernel_3_layer_1_24924 import NNDC
        model = NNDC()
        model.load_state_dict(torch.load('./start_model/start_weights_kernel_3_layer_1_24924.pt'))
    elif attributes['structure'] == 'kernel_3_layer_5':
        if attributes['dropout_rate'] != 0.0:
            print("Wrong choice for Dropout")
            Error = "This combination is not included."
            return Error
        else:
            from models.kernel_3_layer_5_25460 import NNDC
            model = NNDC()
        model.load_state_dict(torch.load('./start_model/start_weights_kernel_3_layer_5_25460.pt'))
    elif attributes['structure'] == 'nearest':
        if attributes['dropout_rate'] == 0.1:
            from models.nearest import NNDC_dropout_10
            model = NNDC_dropout_10()
        elif attributes['dropout_rate'] == 0.05:
            from models.nearest import NNDC_dropout_5
            model = NNDC_dropout_5()
        elif attributes['dropout_rate'] == 0.0:
            from models.nearest import NNDC
            model = NNDC()
        else:
            return "Error: The dropout_rate is not valid"
        from models.nearest import NNDC
        model.load_state_dict(torch.load('./start_model/start_weights_nearest.pt'))
    elif attributes['structure'] == 'convtranspose':
        if attributes['dropout_rate'] == 0.1:
            from models.convtranspose import NNDC_dropout_10
            model = NNDC_dropout_10()
        elif attributes['dropout_rate'] == 0.05:
            from models.convtranspose import NNDC_dropout_5
            model = NNDC_dropout_5()
        elif attributes['dropout_rate'] == 0.0:
            from models.convtranspose import NNDC
            model = NNDC()
        else:
            return "Error: The dropout_rate is not valid"
        from models.convtranspose import NNDC
        model.load_state_dict(torch.load('./start_model/start_weights_convtranspose.pt'))
    # else:
    #     Error = "This combination is not included."


    epochs = 2000
    lr_ = attributes['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr_)
    csx, csy = 5.0, 5.0
    ncx, ncy = 200.0 , 25.0
    npad = 7
    hx_ = [(csx, npad, -1.5), (csx, ncx), (csx, npad, 1.5)]
    hy_ = [(csy, npad, -1.5), (csy, ncy)]
    mesh = TensorMesh([hx_, hy_], x0="CN")
        # files to work with
    dir_path = './' + attributes['Case']
    topo_filename = dir_path + "/topo_xyz.txt"
    data_filename = dir_path + "/dc_data.obs"

    topo_xyz = np.loadtxt(str(topo_filename))
    dc_data = read_dcip2d_ubc(data_filename, "volt", "general")
    dc_data.standard_deviation = attributes['uncertainty'] * np.abs(dc_data.dobs)

    topo_2d = np.unique(topo_xyz[:, [0, 2]], axis=0)
    ind_active = active_from_xyz(mesh, topo_2d)
    survey = dc_data.survey
    survey.drape_electrodes_on_topography(mesh, ind_active, option="top")

    background_conductivity = np.log(1e-2)

        # Define mapping from model to active cells
    nC = int(ind_active.sum()) # Number of cells below the surface<= ind_active.shape[0]
    conductivity_map = maps.IdentityMap(mesh) * maps.ExpMap()
    simulation = dc.simulation_2d.Simulation2DNodal(
        mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True
    )

    W = sdiag(1 / (dc_data.standard_deviation))

    def theta_m(output):
        theta_m = torch.abs(torch.sub(output.flatten(), torch.ones(output.flatten().size(dim=0)), alpha = -4.6))
        return torch.sum(theta_m)

    def total_loss(output, J, beta):
        """
            Note beta here differs from the beta in the convential EM inversion where beta typically starts with a value greater than 1 (e.x. 10), and then cooling down with a fixed rate.
            beta: [0,1]
            total_loss = (1-beta)*theta_phi + beta*theta_m
        """
        theta_phi = torch.matmul(torch.from_numpy(J).type(torch.float32),output.flatten())
        theta_m_ = theta_m(output)
        return (1-beta)*theta_phi + beta*theta_m_

    def decay(x, tau):
        return np.exp(-x/tau)
    diff_list = []
    loss_list = []
    beta_list = []
    reg_list = []
    m_list = []
    tau_ = attributes['decay_rate']
    for epoch in range(epochs):
        beta = decay(epoch, tau_)
        output = model(input_z)
        m = output.detach().numpy()[0][0].flatten()
        print(np.max(m), np.min(m))
	simulation = dc.simulation_2d.Simulation2DNodal(mesh, survey=survey, sigmaMap=conductivity_map, solver=Solver, storeJ=True)
        J = simulation.Jtvec(m, W.T*(W*simulation.residual(m, dc_data.dobs)))
        loss = total_loss(output, J, beta)
        reg = theta_m(output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        diff = W*simulation.residual(m, dc_data.dobs)
        diff = 0.5*np.vdot(diff,diff)
        print("The ",epoch," epoch diff is ", diff)
        print("The ",epoch," total loss is ", loss)
        print("The ",epoch," beta is ", beta)
        print("The ",epoch," reg is ", reg)
        diff_list.append(diff)
        loss_list.append(loss.detach().numpy())
        beta_list.append(beta)
        reg_list.append(reg.detach().numpy())
        m_list.append(m)
    pkl_name = attributes['file_name']+'.pkl'
    f = open(pkl_name, 'wb')
    pickle.dump(m, f)
    pickle.dump(diff_list,f)
    pickle.dump(loss_list,f)
    pickle.dump(beta_list,f)
    pickle.dump(reg_list,f)
    pickle.dump(m_list,f)
    f.close()

    torch.save(model.state_dict(), attributes['file_name']+'_weights.pt')
    torch.save(model, attributes['file_name']+'_model.pt')

    #Return the final result
    model_= NNDC()
    model_.load_state_dict(torch.load(attributes['file_name']+'_weights.pt'))
    output = model_(input_z)
    m = output.detach().numpy()[0][0].flatten()
    pkl_name = attributes['file_name']+'_final'+'.pkl'
    f = open(pkl_name, 'wb')
    pickle.dump(m, f)
    f.close()


if __name__ == '__main__':
    from time import time

    torch.manual_seed(0)
    np.random.seed(0)
    input_z = torch.normal(0, 10, size=(1, 8))
    attributes = {'structure':'kernel_3',
                  'dropout_rate':0.1, #0.05, 0
                  'Case':'PGI_15_Gaussion',
                  'decay_rate':1000,
                  'lr':1e-4,
                  'file_name':'test_case_1.1',
                  'uncertainty':0.05
                  }
    start_time = time()
    train(attributes)
    end_time = time()
    print(end_time - start_time)

