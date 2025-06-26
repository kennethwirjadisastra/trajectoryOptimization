import numpy as np
import matplotlib.pyplot as plt

def plot_traj(V_records, target, alphas=None, ax=None, figsize=(8,6)):
    num_records = V_records.shape[0]

    if alphas is None:
        alphas = np.ones(num_records)
        alphas[:-1] = np.linspace(0.1, 0.5, num_records-1)
    
    arrowDir = lambda V: (V[3]*np.cos(V[2]), V[3]*np.sin(V[2]))

    fig_exists = False

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        fig_exists = True
    
    ax.scatter(V_records[0,0,0], V_records[0,0,1], marker='*', color='sandybrown', s=200, zorder=6, label='Start')
    ax.scatter(target[0], target[1], marker='s', color='sandybrown', s=120, label='Target')
    ax.scatter(V_records[-1,-1,0], V_records[-1,-1,1], marker='o', color='dodgerblue', s=120, zorder=10, label='End')


    ax.arrow(*V_records[0,0,0:2], *arrowDir(V_records[0,0]), color='sandybrown', head_width=0.15, head_length=0.15, lw=4, zorder=2)
    ax.arrow(*target[0:2], *arrowDir(target.numpy()), color='sandybrown', head_width=0.15, head_length=0.15, lw=4, zorder=2)
    ax.arrow(*V_records[-1,-1,0:2], *arrowDir(V_records[-1,-1]), color='dodgerblue', head_width=0.1, head_length=0.1, lw=3, zorder=2)

    for i in range(num_records):
        lw = 1 if alphas[i] != 1 else 2
        ax.plot(V_records[i,:,0], V_records[i,:,1], color='dodgerblue', lw=lw, alpha=alphas[i])

    ax.set_title('Trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.axis('equal')
    ax.grid()
    ax.legend()

    return (fig, ax) if fig_exists is True else ax


def plot_forcing(F_records, ts_records, alphas=None, ax=None, figsize=(8,6)):
    num_records = F_records.shape[0]

    if alphas is None:
        alphas = np.ones(num_records)
        alphas[:-1] = np.linspace(0.1, 0.5, num_records-1)
    
    fig_exists = False
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        fig_exists = True

    for i in range(num_records-1):
        lw = 1.5 if alphas[i] != 1 else 2
        plt.plot(ts_records[i,:-1].detach().numpy(), F_records[i,:,0].detach().numpy(), color='deeppink', lw=lw, alpha=alphas[i])
        plt.plot(ts_records[i,:-1].detach().numpy(), F_records[i,:,1].detach().numpy(), color='limegreen', lw=lw, alpha=alphas[i])
    
    lw = 1 if alphas[-1] != 1 else 2
    plt.plot(ts_records[-1,:-1].detach().numpy(), F_records[-1,:,0].detach().numpy(), color='deeppink', lw=lw, alpha=alphas[-1], label='Acceleration Force')
    plt.plot(ts_records[-1,:-1].detach().numpy(), F_records[-1,:,1].detach().numpy(), color='limegreen', lw=lw, alpha=alphas[-1], label='Turning Force')

    left, right = ax.get_xlim()
    ax.plot([left, right], [0.8, 0.8], '--k', lw=1.5)
    ax.plot([left, right], [-0.8, -0.8], '--k', lw=1.5)
    
    ax.legend()
    ax.set_title('Forcing Terms')
    ax.set_xlabel('Time')
    ax.set_ylabel('Force')
    ax.set_ylim([-0.85, 0.85])
    ax.grid()

    return (fig, ax) if fig_exists is True else ax