import torch


def jitter(A, eps=1e-8):
    jitt = eps * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return A + jitt


def howI(A):
    return torch.max(torch.abs(A - torch.eye(A.shape[0], dtype=A.dtype))).item()


def howL(A):
    return torch.max(torch.abs(torch.triu(A, diagonal=1))).item()


def cholesky_rankone_update_recursive(L, v, b):
    """
    Recursive Cholesky rank-one update. Only applicable
    up to 999 by 999 L matrices (maximum recursion depth).
    Only here for reference, not used in SGP code.
    """
    assert len(L.shape) == 2 and len(v.shape) == 2
    assert L.shape[0] == L.shape[1]
    if L.shape[0] == 1:
        return torch.sqrt(L.pow(2) + b * v.pow(2))
    else:
        # update scalar and vector
        l11 = L[0, 0].unsqueeze(-1).unsqueeze(-1)
        l21 = L[1:, [0]]
        newl11 = torch.sqrt(l11.pow(2) + b * v[0].pow(2))
        newl21 = (L[0, 0] * l21 + b * v[0, :] * v[1:, :]) / newl11
        # recursively update bottom right block
        sub_b = b * (L[0, 0] / newl11).pow(2)
        sub_v = v[1:, :] - l21 * v[0, :] / L[0, 0]
        sub_L = cholesky_rankone_update(L[1:, 1:], sub_v, sub_b)
        # assemble
        upper_L = torch.cat([newl11, torch.zeros(newl21.T.shape, dtype=L.dtype)], dim=1)
        lower_L = torch.cat([newl21, sub_L], dim=1)
        return torch.cat([upper_L, lower_L], dim=0)


def cholesky_rankone_update(L, v, beta=1):
    """
    Iterative Cholesky rank-one update that updates L in place.
    """
    assert len(L.shape) == 2 and len(v.shape) == 1
    assert L.shape[0] == L.shape[1]
    w = v.clone()
    b = 1
    for j in range(len(v)):
        ljj = L[j, j].clone()
        wj = w[j].clone()
        # scalar updates
        L[j, j] = torch.sqrt(ljj.pow(2) + (beta / b) * wj.pow(2))
        gamma = (ljj.pow(2) * b + beta * wj.pow(2)).item()
        # vector updates
        w[(j + 1):] = w[(j + 1):] - (wj / ljj) * L[(j + 1):, j]
        L[(j + 1):, j] = (L[j, j] / ljj) * L[(j + 1):, j] + (L[j, j] * beta * wj / gamma) * w[(j + 1):]
        # update scalar b
        b += (beta * w[j].pow(2) / ljj.pow(2)).item()
    return L


def cholesky_update(L, V, beta=1):
    """
    Iteratively apply rank-one Cholesky updates to update L in place.
    """
    for i in range(V.shape[1]):
        L = cholesky_rankone_update(L, V[:, i], beta)
    return L
