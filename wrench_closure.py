import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

def solve_minimum_wrench(tip_poses, contact_normals):
    """
    tip_poses: (N, 3)
    contact_normals: (N, 3)
    """
    center = tip_poses.mean(dim=0)
    r = tip_poses - center
    Rs = []
    ns = []
    Rp = []
    F = cp.Variable((r.shape[0],3))
    for i in range(r.shape[0]):
        Rs.append(vector_to_skew_symmetric_matrix(r[i]))
        ns.append(cp.Parameter((1,3)))
        Rp.append(cp.Parameter((3,3)))
    R = torch.stack(Rs)

    constraints = [ns[i] @ F[i] >= 0.5 * cp.pnorm(F[i]) for i in range(r.shape[0])]
    constraints = [ns[i] @ F[i] >= 5 for i in range(r.shape[0])] + constraints

    objective = cp.Minimize(0.5 * cp.pnorm(F[0]+F[1]+F[2]+F[3], p=2) + 
                            0.5 * cp.pnorm(Rp[0]@F[0]+Rp[1]@F[1]+Rp[2]@F[2]+Rp[3]@F[3], p=2))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()
    layer = CvxpyLayer(problem, parameters=ns+Rp, variables=[F])
    contact_normals = contact_normals.view(r.shape[0],1,3)
    solution = layer(*contact_normals, *R, solver_args={"eps": 1e-8})[0]
    return solution


# TODO: Should be differentiable
def vector_to_skew_symmetric_matrix(v):
    return torch.tensor([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], requires_grad=v.requires_grad)

if __name__ == "__main__":
    tip_pose = torch.tensor([[0.0,0.0,0.0],
                        [1.0,0.0,0.0],
                        [0.0,1.0,0.0],
                        [0.0,0.0,1.0]]).double()
    center = tip_pose.mean(dim=0)
    n1 = center -torch.tensor([[0.0,0.0,0.0]])
    n2 = center - torch.tensor([[1.0,0.0,0.0]])
    n3 = center - torch.tensor([[0.0,1.0,0.0]])
    n4 = center - torch.tensor([[0.0,0.0,1.0]])
    n1 = (n1 / torch.norm(n1)).double().requires_grad_(True)
    n2 = (n2 / torch.norm(n2)).double().requires_grad_(True)
    n3 = (n3 / torch.norm(n3)).double().requires_grad_(True)
    n4 = (n4 / torch.norm(n4)).double().requires_grad_(True)
    n = torch.stack([n1,n2,n3,n4])

    tip_pose = tip_pose.requires_grad_(True)


    f = solve_minimum_wrench(tip_pose, n)

    f.sum().backward()
    print(f.shape)

    print(f.sum(dim=0), n1.grad)

