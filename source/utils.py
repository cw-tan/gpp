import torch


def howI(A):
    return torch.max(torch.abs(A - torch.eye(A.shape[0], dtype=A.dtype))).item()


def howL(A):
    return torch.max(torch.abs(torch.triu(A, diagonal=1))).item()


def UUT_inv_root_decomp(A):
    # get root decomposition of Psi
    L, Q = torch.linalg.eigh(A)  # O(M'³)
    regularizedQ = Q * torch.sign(L).unsqueeze(-2)
    regularizedL = torch.diag(torch.clamp(torch.abs(L), min=1e-7).pow(-1 / 2))
    aux = regularizedQ @ regularizedL
    # RQ decomposition to make U_Psi upper triangular
    P1 = torch.fliplr(torch.eye(aux.shape[0], dtype=torch.float64))
    P2 = torch.fliplr(torch.eye(aux.shape[1], dtype=torch.float64))
    _, R = torch.linalg.qr((P1 @ aux).T, mode='complete')
    R = P1 @ R.T @ P2
    return R


def UUT_root_decomp(A):
    # get root decomposition of Psi
    L, Q = torch.linalg.eigh(A)  # O(M'³)
    regularizedQT = Q.T * torch.sign(L).unsqueeze(-2)
    regularizedL = torch.diag(torch.abs(L).pow(1 / 2))
    aux = regularizedQT @ regularizedL
    # RQ decomposition to make U_Psi upper triangular
    P1 = torch.fliplr(torch.eye(aux.shape[0], dtype=torch.float64))
    P2 = torch.fliplr(torch.eye(aux.shape[1], dtype=torch.float64))
    _, R = torch.linalg.qr((P1 @ aux).T, mode='complete')
    R = P1 @ R.T @ P2
    return R
