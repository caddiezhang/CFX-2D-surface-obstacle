import numpy as np
import scipy.io as sio
import h5py
from scipy.integrate import simps
import matplotlib.pyplot as plt
# plotting environ
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['font.family'] = 'serif'
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
figl = 11.2
figh = 1.8

if __name__ == '__main__':
    model_index = input('1: sessile 2: runback 3: hemisphere')
    lamturb_index = input('1: laminar 2: turbulent')
    submergence = input('specify relative submergence:')
    # some constants
    rho = 1.225
    mu = 1.8 * 1e-5
    if model_index=='1':
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
        l = 14.45 * 1e-3
        xoff = 0.4430159
    if lamturb_index == '1':
        lamturb = 'lam'
    elif lamturb_index == '2':
        lamturb = 'turb'

    Uinf = 4
    # load velocity fields
    folder = r'C:\Users\caddiezhang\OneDrive - University of Waterloo\11a. Fluent simulation' \
             + r'\flow_over_2d_bump_0212\matlab_files/'
    fname = model + '_' + lamturb +  '_dh' +  submergence + '_cartesian.mat'
    fstat = sio.loadmat(folder + fname)
    xx = fstat['xx'][:] - xoff
    yy = fstat['yy'][:]
    u = fstat['u'][:]
    v = fstat['v'][:]
    p = fstat['p'][:]

    # Set CV boundaries
    cv_xhu = -l / h
    idx_xhu = np.argmin(np.abs(xx[0, :] / h - cv_xhu))
    print(idx_xhu)
    cv_yht = 3
    idx_yht = np.argmin(np.abs(yy[:, 0] / h - cv_yht))
    print(idx_yht)
    cv_xhd_min = 0
    cv_xhd_max = 20
    idx_xhd_min = np.argmin(np.abs(xx[0, :] / h - cv_xhd_min))
    idx_xhd_max = np.argmin(np.abs(xx[0, :] / h - cv_xhd_max))
    print(idx_xhd_min, idx_xhd_max)
    cv_yhb = 0
    idx_yhb = np.argmin(np.abs(yy[:, 0] / h))

    CD_CLs = []
    xhds = []
    for dd, idx_xhd in enumerate(np.arange(idx_xhd_min, idx_xhd_max + 1)):
        print(idx_xhu, idx_xhd)
        xhds.append(xx[0, idx_xhd])
        # get velocity and pressure data on cv boundaries
        # upstream cv boundary
        uavg_xhu = u[idx_yhb:idx_yht, idx_xhu]
        pavg_xhu = p[idx_yhb:idx_yht, idx_xhu]

        # downstream cv boundary
        uavg_xhd = u[idx_yhb:idx_yht, idx_xhd]
        pavg_xhd = p[idx_yhb:idx_yht, idx_xhd]

        # top cv boundary
        uavg_yht = u[idx_yht, idx_xhu:idx_xhd]
        vavg_yht = v[idx_yht, idx_xhu:idx_xhd]

        # surface integrals
        int_xhu = simps(pavg_xhu, yy[idx_yhb:idx_yht, idx_xhu]) \
                  + simps(rho * uavg_xhu ** 2, yy[idx_yhb:idx_yht, idx_xhu])
        int_yht = -simps(rho * uavg_yht * vavg_yht, xx[idx_yht, idx_xhu:idx_xhd])
        int_xhd = -simps(pavg_xhd, yy[idx_yhb:idx_yht, idx_xhd]) \
                  - simps(rho * uavg_xhd ** 2, yy[idx_yhb:idx_yht, idx_xhd])

        # calculate drag
        FD = int_xhu + int_yht + int_xhd
        CD_CLs.append(FD / (1 / 2 * rho * Uinf ** 2 * h))

    CD_CLs = np.array(CD_CLs)
    xhds = np.array(xhds)

    CD_CL = np.mean(CD_CLs)
    CD_CL_err1 = np.min(CD_CLs) - CD_CL
    CD_CL_err2 = np.max(CD_CLs) - CD_CL

    print(CD_CL, CD_CL_err1, CD_CL_err2)

    fig_folder = r'C:\Users\caddiezhang\OneDrive - University of Waterloo\5. Meetings\Weekly Suncor Meetings' \
                 + r'\13. Winter 2020\20200220\figs/'

    fig = plt.figure(figsize = (7.2, 3.6))
    plt.scatter(xhds/h, CD_CLs)
    plt.xlabel(r'$x_d/h$', fontsize = 20)
    plt.ylabel(r'$C_{D_{CL}}$', fontsize = 20)
    plt.title(model+'-'+lamturb+'-dh'+submergence+'-2D simulation', fontsize = 16)
    plt.tight_layout()
    plt.show()
    fig_name = 'CDCL_' + model + '_' + lamturb + '_dh' + submergence + '.png'
    fig.savefig(fig_folder + fig_name, dpi = 150)

    p[u == 0] = np.nan
    fig = plt.figure(figsize = (figl, figh))
    ctf = plt.contourf(xx/h, yy/h, p / (0.5*rho*Uinf**2), levels=np.linspace(-0.4, 0.4, 20), cmap='seismic', extend='both')
    plt.xlim([-20, 50])
    plt.ylim([0.05, 7.5])
    cpos = fig.add_axes([0.65, 0.82, 0.3, 0.03])
    cbar = plt.colorbar(ctf, ticks=np.linspace(-0.4, 0.4, 3), orientation='horizontal', cax=cpos)
    fig.text(0.38, 0.8, r'$\overline{p}/\frac{1}{2} \rho U^2_\infty$', fontsize=28)
    fig.tight_layout()
    plt.show()
    fig_name = 'pavg_' + model + '_' + lamturb + '_dh' + submergence + '.png'
    fig.savefig(fig_folder + fig_name, dpi=150)

    v[u == 0] = np.nan
    fig = plt.figure(figsize = (figl, figh))
    ctf = plt.contourf(xx/h, yy/h, v / Uinf, levels=np.linspace(-0.3, 0.3, 20), cmap='RdBu_r', extend='both')
    plt.xlim([-20, 50])
    plt.ylim([0.05, 7.5])
    cpos = fig.add_axes([0.65, 0.82, 0.3, 0.03])
    cbar = plt.colorbar(ctf, ticks=np.linspace(-0.3, 0.3, 3), orientation='horizontal', cax=cpos)
    fig.text(0.46, 0.8, r'$\overline{v}/U_\infty$', fontsize=28)
    fig.tight_layout()
    plt.show()
    fig_name = 'vavg_' + model + '_' + lamturb + '_dh' + submergence + '.png'
    fig.savefig(fig_folder + fig_name, dpi=150)

    u[u==0] = np.nan
    fig = plt.figure(figsize = (figl, figh))
    ctf = plt.contourf(xx/h, yy/h, u/Uinf, levels = np.linspace(-0.1, 1.2, 20), cmap = 'RdBu_r', extend = 'both')
    #plt.streamplot(xx[0, 0:1000] / h, yy[:21, 0] / h, u[:21, 0:1000] / Uinf, v[:21, 0:1000] / Uinf,
    #                  density=[2, 0.7], linewidth=1, color='k')
    #plt.streamplot(xx[0, 1000:] / h, yy[:21, 0] / h, u[:21, 1000:] / Uinf, v[:21, 1000:] / Uinf, density=[2, 0.7],
    #                  linewidth=1, color='k')
    #plt.streamplot(xx[0, :] / h, yy[20:, 0] / h, u[20:, :] / Uinf, v[20:, :] / Uinf, density=[4, 1],
    #                  linewidth=1, color='k')
    plt.contour(xx / h, yy / h, u / Uinf, levels=np.array([-1e-10]),
                   linestyles='dashed', linewidths=2, colors='white')
    plt.xlim([-20, 50])
    plt.ylim([0.05, 7.5])
    cpos = fig.add_axes([0.65, 0.82, 0.3, 0.03])
    cbar = plt.colorbar(ctf, ticks=np.linspace(-0., 1.0, 3), orientation='horizontal', cax=cpos)
    fig.text(0.46, 0.8, r'$\overline{u}/U_\infty$', fontsize=28)
    fig.tight_layout()
    plt.show()
    fig_name = 'uavg_' + model + '_' + lamturb + '_dh' + submergence + '.png'
    fig.savefig(fig_folder + fig_name, dpi=150)