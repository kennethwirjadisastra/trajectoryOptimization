import torch as pt
import numpy as np
import matplotlib.pyplot as plt

def cat_ts(ts_mat):
    """
    Concatenates segment timesteps for each iteration
    ts_mat: list of lists (n_logs x n_segments)
            each sublist holds n_segments tensors
            each tensor holds times for each segment (n_steps+1)
            may have variable size depending on n_steps for each segment
    """
    num_logs = len(ts_mat)
    num_segments = len(ts_mat[0])
    total_steps = pt.tensor([len(ts_mat[0][n]) for n in range(num_segments)]).sum() - num_segments    
    
    ts_cat = pt.zeros(num_logs, total_steps+1)

    for i in range(num_logs):        
        ts_path = [ts_mat[i][0]]

        prev_time = ts_path[0][-1].item()

        for j in range(1, num_segments):
            ts_path.append(ts_mat[i][j][1:] + prev_time)
            prev_time = ts_path[j][-1]
        
        ts_cat[i,:] = pt.cat(ts_path)
    
    return ts_cat



def cat_F(F_true):
    """
    Concatenates segment forcings for each iteration
    F_true: list of lists (num_logs x n_segments)
            each sublist holds n_segments tensors
            each tensor holds phi, psi, and dt for each segment (n_steps x 3) variable size depending on n_steps
    """
    return pt.stack([pt.cat(F_true[i]) for i in range(len(F_true))], dim=0)


def color_map(n_segments, cmap_name='viridis', vmin=0, vmax=1):
    """
    returns list of n_segments colors
    """
    cmap = plt.get_cmap(cmap_name)
    return [cmap((i / max(n_segments - 1, 1)) * (vmax-vmin) + vmin) for i in range(n_segments)]

def plot_multi_traj(V_mat, target, checkpoint_2d=None, checkpoint_dof=None, dof_init=None, 
              ax=None, alphas=None, colors=None, cmap='viridis', vmin=0, vmax=1, figsize=(8,6)):
    """
    Plots trajectories of car on existing axes object or creates new one
    V_mat: list of lists (n_logs x n_segments)
           each sublist holds n_segments 5D tensors
           each 5D tensor holds position sequence of each segment for a given iteration
    target: target position in 5D space
    checkpoint_2d: tensor of 2D checkpoints (n_checkpoints x 2) (optional)
    checkpoint_dof: tensor of 3D checkpoint degrees of freedom theta, s omega (n_checkpoints x 3) (optional)
    dof_init: intitializations of checkpoint_dof (optional)
    """
    colors = color_map(len(V_mat[0]), cmap_name=cmap, vmin=vmin, vmax=vmax) if colors is None else colors
    if alphas is None:
        alphas = np.ones(len(V_mat))
        alphas[:-1] = np.linspace(0.1, 0.8, len(V_mat)-1)
        
    arrowDir = lambda V: (V[3]*np.cos(V[2]), V[3]*np.sin(V[2]))

    full_checkpoint = pt.cat([checkpoint_2d, checkpoint_dof], dim=1).detach() if checkpoint_2d is not None else None

    fig_exists = False

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        fig_exists = True


    ax.scatter(*V_mat[0][0][0,0:2], marker='*', color='sandybrown', s=200, label='Start')
    ax.arrow(*V_mat[0][0][0,0:2], *arrowDir(V_mat[0][0][0,:]), head_width=0.15, head_length=0.15, lw=4, color='sandybrown', zorder=6)

    if checkpoint_2d is not None:
        ax.scatter(checkpoint_2d[:,0], checkpoint_2d[:,1], marker='X', color='sandybrown', s=240, lw=0.8, label='Checkpoint')


    for i in range(len(V_mat)):
        lw = 2 if alphas[i] == 1 else 1.5
        # plot first segment
        ax.plot(V_mat[i][0][:,0], V_mat[i][0][:,1], lw=lw, color=colors[0], alpha=alphas[i], zorder=4)
        if i == len(V_mat)-1:
            if checkpoint_2d is not None:
                ax.scatter(*V_mat[i][0][-1,0:2], marker='x', color=colors[0], s=160, lw=2, zorder=6)
            ax.arrow(*V_mat[i][0][-1,0:2], *arrowDir(V_mat[i][0][-1,:]), head_width=0.1, head_length=0.1, lw=3, color=colors[0], alpha=alphas[i], zorder=6)

        # plot subsequent segments
        for j in range(len(V_mat[i])-1):
            ax.plot(V_mat[i][j+1][:,0], V_mat[i][j+1][:,1], lw=lw, color=colors[j+1], alpha=alphas[i], zorder=4)
            if i == len(V_mat)-1:
                if full_checkpoint is not None:
                    ax.arrow(*checkpoint_2d[j], *arrowDir(full_checkpoint[j]), head_width=0.15, head_length=0.15, lw=4, color='sandybrown', zorder=5)
                if j != len(V_mat[i])-2:
                    ax.scatter(*V_mat[i][j+1][-1,0:2], marker='x', color=colors[j+1], s=160, lw=2, zorder=6)
                ax.arrow(*V_mat[i][j+1][-1,0:2], *arrowDir(V_mat[i][j+1][-1,:]), head_width=0.1, head_length=0.1, lw=3, color=colors[j+1], alpha=alphas[i], zorder=6)
                if dof_init is not None:
                    ax.arrow(*checkpoint_2d[j], dof_init[j,1]*np.cos(dof_init[j,0]), dof_init[j,1]*np.sin(dof_init[j,0]),
                             head_width=0.15, head_length=0.15, lw=4, color='black', zorder=5, label='Initial Guess')



    ax.scatter(*target[0:2],  marker='s', color='sandybrown', s=120, label='Target')
    ax.arrow(*target[0:2], *arrowDir(target), head_width=0.15, head_length=0.15, lw=4, color='sandybrown', zorder=5)
    ax.scatter(*V_mat[-1][-1][-1,0:2], marker='D', color=colors[-1], s=120, label='End', zorder=6)
    ax.arrow(*V_mat[-1][-1][-1,0:2], *arrowDir(V_mat[-1][-1][-1,:]), head_width=0.1, head_length=0.1, lw=3, color=colors[-1], zorder=6)


    # sort legend labels
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    order = [0, 1, 3, 2, 4] if checkpoint_2d is not None else range(3)
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    ordered_handle_list = []
    ordered_label_list = []
    [ordered_handle_list.append(handle_list[idx]) for idx in order if handle_list[idx] not in ordered_handle_list]
    [ordered_label_list.append(label_list[idx]) for idx in order if label_list[idx] not in ordered_label_list]

    ax.legend(ordered_handle_list, ordered_label_list)#, labelspacing=1.5, fontsize=14)
    ax.set_title('Car Trajecotry')#, fontsize=16)
    ax.set_xlabel('X Position')#, fontsize=14)
    ax.set_ylabel('Y Position')#, fontsize=14)
    ax.grid()
    return (fig, ax) if fig_exists is True else ax


def plot_multi_forcing(F_cat, ts_cat, n_steps, ax=None, alphas=None, colors=None, figsize=(8,6)):
    """
    Plots evolution of forcing terms on existin axes object or creates a new one
    F_cat: tensor of concatenated true forcings (n_logs x total_steps x 3)
    ts_cat: tensor of times for each log (n_logs x total_steps)
    n_stesp: number of timesteps for each segment
    """
    alphas = np.linspace(0.1, 0.8, F_cat.shape[0]-1) if alphas is None else alphas
    colors = ['deeppink', 'limegreen'] if colors is None else colors

    fig_exists = False

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        fig_exists = True
    
    for i in range(ts_cat.shape[0]-1):
        lw = 2 if alphas[i] == 1 else 1.5
        ax.plot(ts_cat[i,:-1], F_cat[i,:,0].detach().numpy(), color=colors[0], alpha=alphas[i], lw=lw, zorder=0)
        ax.plot(ts_cat[i,:-1], F_cat[i,:,1].detach().numpy(), color=colors[1], alpha=alphas[i], lw=lw, zorder=1)

    ax.plot(ts_cat[-1,:-1], F_cat[-1,:,0].detach().numpy(), color=colors[0], alpha=1, lw=2, zorder=0, label='Acceleration Force')
    ax.plot(ts_cat[-1,:-1], F_cat[-1,:,1].detach().numpy(), color=colors[1], alpha=1, lw=2, zorder=1, label='Turning Force')

    left, right = ax.get_xlim()
    ax.plot([left, right], [0.8, 0.8], '--k', lw=2)
    ax.plot([left, right], [-0.8, -0.8], '--k', lw=2)

    steps_taken = 0
    for step in n_steps:
        steps_taken += step
        ax.plot(2*[ts_cat[-1, steps_taken]], [-0.8, 0.8], '--k', lw=2)


    ax.set_title('Forcing Inputs')#, fontsize=16)
    ax.set_xlabel('Time')#, fontsize=14)
    ax.set_ylabel('Force')#, fontsize=14)
    ax.set_xlim(left, right)
    ax.grid()
    ax.legend()#fontsize=14)

    return (fig, ax) if fig_exists is True else ax