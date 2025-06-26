import torch as pt
import numpy as np
import numbers

def V_step(V, F, dt=None):
    """
    Computes next position given current position and forcing inputs
    V: car's current position in 6D space, (x , y, theta, s, omega, t)
    F: current forcing inputs, (phi, psi, dt)
    """
    phi = F[0]  # linear acceleration
    psi = F[1]  # angluar acceleration
    dt = F[2] if dt is None else dt   # timestep size
    dt2 = dt*dt/2

    x = V[0]    # x coordinate
    y = V[1]    # y coordinate
    a = V[2]    # angle
    s = V[3]    # speed
    w = V[4]    # angular speed
    t = V[5]    # time

    _0 = pt.zeros_like(x)
    _1 = pt.ones_like(x)

    A1 = pt.stack([x, y, a, s, w, t])
    A2 = pt.stack([s*pt.cos(a), s*pt.sin(a), w, _0, _0, _0])
    A3 = pt.stack([-s*w*pt.sin(a), s*w*pt.cos(a), _0, _0, _0, _0])
    A = A1 + dt*A2 + dt2*A3

    B2 = pt.stack([_0, _0, _0, _1, _0, _0])
    B3 = pt.stack([pt.cos(a), pt.sin(a), _0, _0, _0, _0])
    B = dt*B2 + dt2*B3

    C2 = pt.stack([_0, _0, _0, _0, _1, _0])
    C3 = pt.stack([_0, _0, _1, _0, _0, _0])
    C = dt*C2 + dt2*C3

    D = pt.stack([_0, _0, _0, _0, _0, _1])  # this is correct

    # V1 = A + phi*B + psi*C + dt*D
    # return pt.clamp(V1,
    #     min=pt.tensor([-pt.inf, -pt.inf, -pt.inf, 0, -2, -pt.inf]),
    #     max=pt.tensor([pt.inf, pt.inf, pt.inf, pt.inf, 2, pt.inf]))

    return A + phi*B + psi*C + dt*D

def angle_diff(theta1, theta2):
    """
    Returns smallest angular difference in (0, pi) between two angles
    """
    diff = theta1 - theta2
    return (diff + np.pi) % (2 * np.pi) - np.pi


def to_tensor(V_mat):
    """
    Splits V_mat into position components (5D) and time component (1D)
    """
    ten = pt.zeros((len(V_mat), V_mat[0].shape[0]))
    for i in range(len(V_mat)):
        ten[i] = V_mat[i].detach().cpu()
    return ten[:,0:5], ten[:,5]

def to_lambda(schedule):
    if schedule is None:
        return lambda x, target_loss: 0
    if isinstance(schedule, numbers.Number):
        return lambda x, target_loss: schedule
    if callable(schedule):
        return schedule
    raise TypeError(f'Invalid schedule type: {type(schedule)}. Expected None, a number, or a function.')