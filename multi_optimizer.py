import torch as pt
from physics import *

def multi_loss_fun(Y, T, F, scale=None):
    """
    Y: final state (6D: x, y, theta, s, omega, t)
    T: target (2D or 5D)
    F: list of forcings
    """
    scale = pt.tensor([1, 1, 1, 1, 1]) if scale is None else scale
    diff = (Y[0:5]-T[0:5])
    diff[2] = angle_diff(Y[2], T[2])
    position_loss = pt.sum(diff * diff * scale)       # loss from missing the target
    time_loss = Y[5]                                # loss by taking too long
    l2_loss = pt.mean(pt.stack(F)[:,0:2]**2)        # loss by using too much force
    return position_loss, time_loss, l2_loss

def get_time_forcing(V, F):
    """
    Returns true forcings from logits
    """
    limits_min = pt.tensor([-0.8, -0.8, 0.001], device=F.device)
    limits_max = pt.tensor([0.8, 0.8, 0.25], device=F.device)
    return limits_min + (limits_max - limits_min) * pt.sigmoid(F)

def compute_multi_path_autograd(V0, F_logits, target):
    """
    Computes entire path of segment given initial position and logits of forcings. Computes loss given target.
    V0: initial position
    F_logits: logits of forcing terms
    target: target position in 5D space
    """
    V_list = [V0]    # trajectory as a list of position vectors
    F_true = []      # forcing terms as a list of forcing vectors

    for i in range(F_logits.shape[0]):
        F_true.append(get_time_forcing(V_list[-1], F_logits[i]))
        V_list.append(V_step(V_list[i], F_true[i]))

    loss = multi_loss_fun(V_list[-1], target, F_true, scale=pt.tensor([1, 1, 1, 1, 1]))

    return V_list, loss


def optimizeMultiPath(v0, target, F_logits, checkpoint_2d=None, checkpoint_dof=None, iters=500, fine_tuning_steps=50, plot_freq=50,
        lr=9e-2, l2_rate=-1e0, time_rate=2e-8, reverse_rate=0.5, show_logs=True
    ):
    """
    Iteratively performs GD to optimize forcings
    
    v0: starting tensor in 6d (v0[5]=0)
    checkpoint_2d: num_checkpoints x 2 tensor containing fixed points in 2d
    target: ending tensor in 5d
    checkpoint_dof: num_checkpoints x 3 tensor containing learnable dfs at start of segments
    F_logits: list (length = number of checkpoints + 1)
                each element is a n_steps x 3 tensor of forcing terms
    """
    lr = to_lambda(lr)
    l2_rate = to_lambda(l2_rate)
    time_rate = to_lambda(time_rate)
    reverse_rate = to_lambda(reverse_rate)

    n_segments = checkpoint_2d.shape[0] + 1 if checkpoint_2d is not None else 1
    n_intermediate_segments = n_segments - 2
    num_logs = iters//plot_freq if iters%plot_freq == 0 else iters//plot_freq + 1

    optim = pt.optim.Adam(F_logits + [checkpoint_dof]) if checkpoint_dof is not None else pt.optim.Adam(F_logits)

    V_mat = []
    ts_mat = []
    F_true = []
    loss_logs = pt.zeros((num_logs, n_segments, 4)) # pos, time, l2, reverse

    for i in range(iters):
        is_log_step = (i % plot_freq == plot_freq - 1) or (i== iters - 1)
        log_idx = i // plot_freq
        
        if is_log_step:
            # log v, ts, and F on current iteration's path
            v_path = []
            ts_path = []
            F_true_path = []

        loss = 0
        x = i / (iters - fine_tuning_steps)

        # compute first segment
        optim.zero_grad()

        end = pt.cat([checkpoint_2d[0], checkpoint_dof[0], pt.zeros(1, device=v0.device)], dim=0) if checkpoint_dof is not None else target
        traj, (position_loss, time_loss, l2_loss) = compute_multi_path_autograd(v0, F_logits[0], end)

        # compute loss
        l2_coef = l2_rate(x, position_loss) if i < iters-fine_tuning_steps else 0
        time_coef = time_rate(x, position_loss) if i < iters-fine_tuning_steps else 0
        reverse_coef = reverse_rate(x, position_loss) if i < iters-fine_tuning_steps else 0
                

        #reverse_loss = pt.mean(pt.clamp(pt.stack(traj)[:,3], max=0.0)**2)
        #reverse_loss = pt.mean(pt.sigmoid(-10 * pt.stack(traj)[:,3]))
        reverse_loss = pt.mean(pt.exp(-5 * pt.stack(traj)[:,3]))

        if is_log_step:
            v_segment, ts_segment = to_tensor(traj)
            v_path.append(v_segment)
            ts_path.append(ts_segment)
            F_true_path.append(get_time_forcing(traj[:-1], F_logits[0][:, :]))
            loss_logs[log_idx,0,:] =  pt.tensor([position_loss, time_loss, l2_loss, reverse_loss])

        if n_segments == 1:
            loss += position_loss + time_coef*time_loss + l2_coef*l2_loss + reverse_coef*reverse_loss
        else:
            loss += 3/2*position_loss + time_coef*time_loss + l2_coef*l2_loss + reverse_coef*reverse_loss

        for j in range(n_intermediate_segments+1):
            start = end
            if j == n_segments-2:
                end = target
            else:
                end = pt.cat([checkpoint_2d[j+1], checkpoint_dof[j+1], pt.zeros(1, device=v0.device)], dim=0)

            traj, (position_loss, time_loss, l2_loss) = compute_multi_path_autograd(start, F_logits[j+1], end)

            #reverse_loss = pt.mean(pt.clamp(pt.stack(traj)[:,3], max=0.0)**2)
            #reverse_loss = pt.mean(pt.sigmoid(-10 * pt.stack(traj)[:,3]))
            reverse_loss = pt.mean(pt.exp(-5 * pt.stack(traj)[:,3]))


            if is_log_step:
                v_segment, ts_segment = to_tensor(traj)
                v_path.append(v_segment)
                ts_path.append(ts_segment)
                F_true_path.append(get_time_forcing(traj[:-1], F_logits[j+1][:, :]))
                loss_logs[log_idx,j+1,:] =  pt.tensor([position_loss, time_loss, l2_loss, reverse_loss])

            if j == n_segments - 2:
                # no coefficient on position_loss because we don't need to enforce continuity as strictly at the end
                loss += position_loss + time_coef*time_loss + l2_coef*l2_loss + reverse_coef*reverse_loss
            else:
                loss += 3/2*position_loss + time_coef*time_loss + l2_coef*l2_loss + reverse_coef*reverse_loss

        if is_log_step:
            V_mat.append(v_path)
            ts_mat.append(ts_path)
            F_true.append(F_true_path)

        loss.backward()

        # set optimizer learning rate and take step
        step_lr = lr(x, position_loss)
        for param_group in optim.param_groups:
            param_group['lr'] = step_lr
        optim.step()


        # plot solution every plot_freq steps
        if show_logs and is_log_step:
            
            print(f'Iteration {i+1}, Mean Position Dist: {loss_logs[log_idx, :, 0].mean() ** 0.5:.3f},  '\
                  f'End Time: {loss_logs[log_idx,:,1].sum():.3f},  '\
                  f'L2: {loss_logs[log_idx,:,2].mean():.3f},  '\
                  f'Reverse Loss: {loss_logs[log_idx,:,3].mean():.3f}')
                   
    return F_logits, F_true, checkpoint_dof, V_mat, ts_mat, loss_logs