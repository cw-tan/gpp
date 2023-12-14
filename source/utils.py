import torch


def jitter(A, eps=1e-8):
    return A + eps * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)


def cholesky_update(L, V):
    """
    Rank-R Cholesky update to compute L' where L' @ L'.T = L @ L.T + V @ V.T
    Implementation is based on Green, D. R. (1968). Algorithms: Algorithm 319:
    Triangular factors of modified matrices. Communications of the ACM, 11(1), 12.
    """
    W = V.clone()
    c = torch.eye(W.shape[1], dtype=W.dtype, device=L.device)
    for i in range(L.shape[0]):
        d = L[i, i].clone()
        p = torch.mv(c, W[i, :])
        L[i, i] = torch.sqrt(d.pow(2) + torch.dot(W[i, :], p))
        p /= L[i, i]
        L[(i + 1):, i] = L[(i + 1):, i] / d
        W[(i + 1):, :] -= torch.outer(L[(i + 1):, i], W[i, :])
        L[(i + 1):, i] = L[i, i] * L[(i + 1):, i] + torch.mv(W[(i + 1):, :], p)
        c -= torch.outer(p, p)
    return L


def howI(A):
    return torch.max(torch.abs(A - torch.eye(A.shape[0], dtype=A.dtype))).item()


def howL(A):
    return torch.max(torch.abs(torch.triu(A, diagonal=1))).item()
