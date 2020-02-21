"""
Read in the flow field over 2D surface-mounted obstacles from CFX simulation.
Plot the pressure distribution over the model surface.
Calculate the model drag by surface pressure integral.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from functions.mask_edge_surface_integral import edge_finder

# plotting environ
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family'] = 'serif'
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
figl = 7.2
figh = 3.6

if __name__ == '__main__':
    # set path
    #model_index = input('1: sessile 2: runback 3: hemisphere')
    #lamturb_index = input('1: laminar 2: turbulent')
    #submergence = input('specify relative submergence:')
    model_index = '3'
    lamturb_index = '2'
    submergence = '0'
    # some constants
    rho = 1.225
    mu = 1.8 * 1e-5
    Uinf = 4
    if model_index == '1':
        model = 'sessile'
        h = 7.69 * 1e-3
        l = 22.41 * 1e-3
        xoff = 0
    elif model_index == '2':
        model = 'runback'
        h = 7.94 * 1e-3
        l = 24.08 * 1e-3
        xoff = 0
    elif model_index == '3':
        model = 'hemisphere'
        h = 7.725 * 1e-3
        l = 16.4 * 1e-3
        xoff = 0.4430159
    if lamturb_index == '1':
        lamturb = 'lam'
    elif lamturb_index == '2':
        lamturb = 'turb'

    folder = r'C:\Users\caddiezhang\OneDrive - University of Waterloo' + \
             r'\11a. Fluent simulation\flow_over_2d_bump_0212\matlab_files/'
    fname = model + '_' + lamturb + '_dh' + submergence + '_cartesian.mat'
    fstat = sio.loadmat(folder + fname)
    xx = fstat['xx'][:] - xoff
    yy = fstat['yy'][:]
    u = fstat['u'][:]
    v = fstat['v'][:]
    p = fstat['p'][:]

    # Compute shear stress
    dy = np.abs(yy[1,0] - yy[0,0])
    _, dudy = np.gradient(u, dy)

    # Set mask and find mask boundaries
    # find mask boundaries
    MASK = np.zeros(np.shape(p))
    MASK[v== 0] = 1
    pixel_offset = 0
    [_, _, _, _, _, _, idx_bd_left, idx_bd_top, idx_bd_right, idx_bd_bottom, idx_bd] = edge_finder(MASK,
                                                                                                   pixel_offset)

    # sort the model surface boundary
    idx_col = np.sort(idx_bd[1])
    idx_row = idx_bd[0][np.argsort(idx_bd[1])]
    x_bd = list([])
    y_bd = list([])
    p_bd = list([])
    dudy_bd = list([])
    for ii in range(len(idx_col)):
        if (xx[idx_row[ii], idx_col[ii]]/h > -l/h and xx[idx_row[ii], idx_col[ii]]/h <0):
            x_bd.append(xx[idx_row[ii], idx_col[ii]])
            y_bd.append(yy[idx_row[ii], idx_col[ii]])
            p_bd.append(p[idx_row[ii], idx_col[ii]])
            dudy_bd.append(dudy[idx_row[ii], idx_col[ii]])
    x_bd = np.array(x_bd)
    y_bd = np.array(y_bd)
    p_bd = np.array(p_bd)
    dudy_bd = np.array(dudy_bd)
    Cp_bd = p_bd / (rho * Uinf ** 2 / 2)
    Ctaux_bd = mu* dudy_bd / (rho * Uinf ** 2 / 2)
    CdCL_pressure = np.sum(((Cp_bd[1:] + Cp_bd[:-1]) / 2 * (y_bd[1:] - y_bd[:-1]) / h)[:])
    CdCL_shear = np.sum(((Ctaux_bd[1:] + Ctaux_bd[:-1]) / 2 * \
                         np.sqrt((x_bd[1:] - x_bd[:-1])**2+(y_bd[1:] - y_bd[:-1])**2) / h)[:])
    CdCL = CdCL_pressure + CdCL_shear
    print('CdCL:', CdCL)

    # Smooth the boundary
    # Calculate window size
    ws = (lambda x, n: int(x / n) + 1 if int(x / n) % 2 == 0 else int(x / n))
    WS_SG = ws(len(y_bd), 3)
    y_bd = savgol_filter(y_bd, WS_SG, 2)
    # Interpolate on remeshed x to remove jumps in boundary
    x_bd_new = np.linspace(np.amin(x_bd), np.amax(x_bd), len(x_bd))
    f_1d = interp1d(x_bd, y_bd)
    y_bd_intp = f_1d(x_bd_new)
    WS_SG_new = ws(len(y_bd_intp), 7)
    y_bd_new = savgol_filter(y_bd_intp, WS_SG_new, 2)

    # Interpolate pressure field on new mesh
    p_bd_new = griddata((xx.flatten(), yy.flatten()), p.flatten(), (x_bd_new, y_bd_new), method='linear')
    p_bd_new[np.isnan(p_bd_new)] = 0
    Cp_bd_new = p_bd_new / (rho * Uinf ** 2 / 2)

    dudy_bd_new = griddata((xx.flatten(), yy.flatten()), dudy.flatten(), (x_bd_new, y_bd_new), method='linear')
    dudy_bd_new[np.isnan(dudy_bd_new)] = 0
    Ctaux_bd_new = mu*dudy_bd_new/ (rho * Uinf ** 2 / 2)


    plt.scatter(x_bd_new, Cp_bd_new + Ctaux_bd_new)
    plt.show()

    CdCL_pressure_new = np.sum(((Cp_bd_new[1:] + Cp_bd_new[:-1])/2*(y_bd_new[1:] - y_bd_new[:-1])/h)[:])
    #CdCL_pressure_new = np.sum((Cp_bd_new[:-1]  * (y_bd_new[1:] - y_bd_new[:-1]) / h)[:])
    CdCL_shear_new = CdCL_shear = np.sum(((Ctaux_bd_new[1:] + Ctaux_bd_new[:-1]) / 2 * \
                         np.sqrt((x_bd_new[1:] - x_bd_new[:-1])**2+(y_bd_new[1:] - y_bd_new[:-1])**2) / h)[:])
    #CdCL_shear_new = CdCL_shear = np.sum((Ctaux_bd_new[:-1] * \
    #                     np.sqrt((x_bd_new[1:] - x_bd_new[:-1]) ** 2 + (y_bd_new[1:] - y_bd_new[:-1]) ** 2) / h)[:])
    CdCL_new = CdCL_pressure_new + CdCL_shear_new
    print('CdCL_new:', CdCL_new)

    # test plot
    p_nan = p[:]
    p_nan[u**2 + v**2 == 0] = np.nan
    fig = plt.figure(figsize=(figl, figh))
    ctf = plt.contourf(xx/h, yy/h, p / (0.5*rho*Uinf**2), levels=np.linspace(-0.4, 0.4, 20), cmap='seismic', extend='both')
    plt.plot(x_bd / h, y_bd / h, color='k', linewidth = 2)
    plt.plot(x_bd_new / h, y_bd_new / h, color='r', linewidth=2)
    plt.xlim([-20, 50])
    plt.ylim([0.05, 7.5])
    plt.show()
