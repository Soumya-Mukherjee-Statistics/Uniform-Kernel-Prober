# import numpy as np
# import torch
# from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
#
# # Set the device for computation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # # CCA
#
# def cca_decomp(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None):
#     """Computes CCA vectors, correlations, and transformed matrices with GPU support."""
#
#     # Eigen decomposition for A and B
#     if evals_a is None or evecs_a is None:
#         evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
#     if evals_b is None or evecs_b is None:
#         evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
#
#     # Clip small negative eigenvalues to zero
#     evals_a = (evals_a + torch.abs(evals_a)) / 2
#     inv_a = torch.sqrt(torch.where(evals_a > 0, 1 / evals_a, torch.tensor(0.0, device=device)))
#
#     evals_b = (evals_b + torch.abs(evals_b)) / 2
#     inv_b = torch.sqrt(torch.where(evals_b > 0, 1 / evals_b, torch.tensor(0.0, device=device)))
#
#     # Covariance matrix computation
#     cov_ab = A @ B.T
#
#     # Inner SVD problem
#     temp = (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ cov_ab @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T)
#     try:
#         u, s, vh = torch.linalg.svd(temp)
#     except:
#         u, s, vh = torch.linalg.svd(temp * 100)
#         s = s / 100
#
#     transformed_a = (u.T @ (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ A).T
#     transformed_b = (vh @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T) @ B).T
#
#     return u, s, vh, transformed_a, transformed_b
#
#
# def mean_sq_cca_corr(rho):
#     """Compute mean squared CCA correlation."""
#     return 1 - torch.sum(rho * rho).item() / len(rho)
#
#
# def mean_cca_corr(rho):
#     """Compute mean CCA correlation."""
#     return 1 - torch.sum(rho).item() / len(rho)
#
#
# def pwcca_dist(A, rho, transformed_a):
#     """Computes projection weighted CCA distance between A and B."""
#
#     in_prod = transformed_a.T @ A.T
#     weights = torch.sum(torch.abs(in_prod), axis=1)
#     weights = weights / torch.sum(weights)
#     dim = min(len(weights), len(rho))
#
#     return 1 - torch.dot(weights[:dim], rho[:dim]).item()
#
#
# # # CKA
#
# def lin_cka_dist(A, B):
#     """Computes Linear CKA distance between representations A and B."""
#
#     similarity = torch.norm(B @ A.T, p='fro') ** 2
#     normalization = torch.norm(A @ A.T, p='fro') * torch.norm(B @ B.T, p='fro')
#
#     return (1 - similarity / normalization).item()
#
#
# def lin_cka_prime_dist(A, B):
#     """Computes Linear CKA prime distance between representations A and B."""
#
#     if A.shape[0] > A.shape[1]:
#         At_A = A.T @ A  # O(n * n * a)
#         Bt_B = B.T @ B  # O(n * n * a)
#         numerator = torch.sum((At_A - Bt_B) ** 2)
#         denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
#         return (numerator / denominator).item()
#     else:
#         similarity = torch.norm(B @ A.T, p='fro') ** 2
#         denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
#         return 1 - 2 * similarity / denominator
#
#
# # # CKA distance using Gaussian or Laplace kernels
#
# def cka_dist(A, B, A_kernelized=None, B_kernelized=None, kerneltype=1, sigma=100):
#     """Computes CKA distance between representations A and B using Gaussian or Laplace kernels."""
#     if A_kernelized is None or B_kernelized is None:
#         if kerneltype == 1:
#             A_kernelized = cuda_rbf_kernel(A.T, sigma=sigma).to(device)
#             B_kernelized = cuda_rbf_kernel(B.T, sigma=sigma).to(device)
#         else:
#             A_kernelized = cuda_laplacian_kernel(A.T, sigma=sigma).to(device)
#             B_kernelized = cuda_laplacian_kernel(B.T, sigma=sigma).to(device)
#
#     similarity = torch.norm(B_kernelized @ A_kernelized.T, p='fro') ** 2
#     normalization = torch.norm(A_kernelized @ A_kernelized.T, p='fro') * torch.norm(B_kernelized @ B_kernelized.T,
#                                                                                     p='fro')
#
#     return (1 - similarity / normalization).item()
#
#
# # # Procrustes
#
# def procrustes(A, B):
#     """Computes Procrustes distance between representations A and B."""
#
#     n = A.shape[1]
#     A_sq_frob = torch.sum(A ** 2) / n
#     B_sq_frob = torch.sum(B ** 2) / n
#     nuc = torch.linalg.norm(A @ B.T, ord='nuc') / n  # O(p * p * n)
#
#     return (A_sq_frob + B_sq_frob - 2 * nuc).item()
#
#
# # # Our predictor distances
#
# def GULP_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
#     """Computes distance between best linear predictors on representations A and B."""
#
#     n = A.shape[1]
#
#     # Compute eigenvalues and eigenvectors if not provided
#     if evals_a is None or evecs_a is None:
#         evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
#     if evals_b is None or evecs_b is None:
#         evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
#
#     if evals_a is None:
#         inv_a_lambda = None
#     else:
#         evals_a = (evals_a + torch.abs(evals_a)) / (2 * n)
#         inv_a_lmbda = torch.where(evals_a > 0, 1 / (evals_a + lmbda), 1 / lmbda)
#
#     if evals_b is None:
#         inv_b_lambda = None
#     else:
#         evals_b = (evals_b + torch.abs(evals_b)) / (2 * n)
#         inv_b_lmbda = torch.where(evals_b > 0, 1 / (evals_b + lmbda), 1 / lmbda)
#
#     if evals_a is None or evecs_a is None:
#         return None
#     else:
#         T1 = torch.sum(torch.square(evals_a * inv_a_lmbda))
#         T2 = torch.sum(torch.square(evals_b * inv_b_lmbda))
#
#         cov_ab = A @ B.T / n
#         T3 = torch.trace(
#             cov_ab.T @ (evecs_a @ torch.diag(inv_a_lmbda) @ evecs_a.T)
#             @ cov_ab
#             @ (evecs_b @ torch.diag(inv_b_lmbda) @ evecs_b.T)
#         )
#
#         return (T1 + T2 - 2 * T3).item()
#
# # # UKP distance
#
# def UKP_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, A_kernelized=None, B_kernelized=None, kerneltype=1, sigma=1, lmbda=0.01):
#     """
#     Computes distance between best kernel ridge regression predictors on representations A and B, based on Gaussian or Laplace kernels.
#
#     Parameters:
#         A (matrix): k by n matrix of representations of first model.
#         B (matrix): l by n matrix of representations of second model.
#         kerneltype (int): type of kernel to use (1 if Gaussian RBF, 2 if Laplace).
#         sigma (float): bandwidth parameter of kernel (same as in rbf_kernel or laplacian_kernel functions in scikit-learn).
#         lmbda (float): regularization parameter.
#
#     Returns:
#         UKP distance between A and B (scalar).
#     """
#
#     n = A.shape[1]
#
#     if A_kernelized is None or B_kernelized is None:
#         if kerneltype == 1:
#             A_kernelized = cuda_rbf_kernel(A.T, sigma=sigma).to(device)
#             B_kernelized = cuda_rbf_kernel(B.T, sigma=sigma).to(device)
#         else:
#             A_kernelized = cuda_laplacian_kernel(A.T, sigma=sigma).to(device)
#             B_kernelized = cuda_laplacian_kernel(B.T, sigma=sigma).to(device)
#
#     if evals_a is None or evecs_a is None:
#         try:
#             evals_a, evecs_a = torch.linalg.eigh(A_kernelized)
#         except torch._C._LinAlgError as e:
#             evals_a = None
#             evecs_a = None
#     if evals_b is None or evecs_b is None:
#         try:
#             evals_b, evecs_b = torch.linalg.eigh(B_kernelized)
#         except torch._C._LinAlgError as e:
#             evals_b = None
#             evecs_b = None
#
#     if evals_a is None:
#         inv_a_lambda = None
#     else:
#         evals_a = (evals_a + torch.abs(evals_a)) / 2
#         inv_a_lmbda = torch.where(evals_a > 0, 1 / (evals_a + n * lmbda), 1 / n * lmbda)
#
#     if evals_b is None:
#         inv_b_lambda = None
#     else:
#         evals_b = (evals_b + torch.abs(evals_b)) / 2
#         inv_b_lmbda = torch.where(evals_b > 0, 1 / (evals_b + n * lmbda), 1 / n * lmbda)
#
#     if evals_a is None or evecs_a is None or evals_b is None or evecs_b is None:
#         return None
#     else:
#         T1 = torch.sum(torch.square(evals_a * inv_a_lmbda))
#         T2 = torch.sum(torch.square(evals_b * inv_b_lmbda))
#
#         T3 = torch.trace(A_kernelized @ (evecs_a @ torch.diag(inv_a_lmbda) @ evecs_a.T)
#                          @ B_kernelized
#                          @ (evecs_b @ torch.diag(inv_b_lmbda) @ evecs_b.T))
#
#         return (T1 + T2 - 2 * T3).item()
#
#
# def squared_procrustes(A, B):
#     """
#     Computes distance between best linear predictors on representations A and B.
#     """
#
#     n = A.shape[1]
#
#     cov_a = A @ A.T / (n - 1)
#     cov_b = B @ B.T / (n - 1)
#     cov_ab = A @ B.T / (n - 1)
#
#     T1 = torch.trace(cov_a @ cov_a)
#     T2 = torch.trace(cov_b @ cov_b)
#     T3 = torch.trace(cov_ab @ cov_ab.T)
#
#     return (T1 + T2 - 2 * T3).item()
#
# # # Defining kernel functions
#
# def cuda_rbf_kernel(X, Y=None, sigma=1.0):
#     """
#     Computes the RBF (Gaussian) kernel between two sets of samples.
#
#     Parameters:
#         X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
#         Y (torch.Tensor): Second input tensor of shape (n_samples_Y, n_features).
#                         If None, the kernel is computed with X itself.
#         sigma (float): Bandwidth parameter for the RBF kernel.
#
#     Returns:
#         torch.Tensor: RBF kernel matrix of shape (n_samples_X, n_samples_Y).
#     """
#     if Y is None:
#         Y = X
#
#     # Ensure both inputs are on the same device (CUDA)
#     if not (X.is_cuda and Y.is_cuda):
#         raise ValueError("Both inputs must be CUDA tensors.")
#
#     # Compute squared Euclidean distance
#     X_sq = torch.sum(X ** 2, dim=1, keepdim=True)  # shape (n_samples_X, 1)
#     if Y is None:
#         Y_sq = X_sq
#     else:
#         Y_sq = torch.sum(Y ** 2, dim=1, keepdim=True)  # shape (n_samples_Y, 1)
#
#     # Compute the squared distance matrix
#     dists = X_sq + Y_sq.t() - 2 * torch.mm(X, Y.t())  # shape (n_samples_X, n_samples_Y)
#
#     # Apply the RBF kernel formula
#     K = torch.exp(-dists / (2 * sigma ** 2))
#
#     return K
#
# def cuda_laplacian_kernel(X, Y=None, sigma=1.0):
#     """
#     Computes the RBF (Gaussian) kernel between two sets of samples.
#
#     Parameters:
#         X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
#         Y (torch.Tensor): Second input tensor of shape (n_samples_Y, n_features).
#                         If None, the kernel is computed with X itself.
#         sigma (float): Bandwidth parameter for the RBF kernel.
#
#     Returns:
#         torch.Tensor: RBF kernel matrix of shape (n_samples_X, n_samples_Y).
#     """
#     if Y is None:
#         Y = X
#
#     # Ensure both inputs are on the same device (CUDA)
#     if not (X.is_cuda and Y.is_cuda):
#         raise ValueError("Both inputs must be CUDA tensors.")
#
#     # Compute the L1 distance matrix
#     dists= torch.cdist(X, Y, p=1)
#
#     # Apply the RBF kernel formula
#     K = torch.exp(-dists / sigma)
#
#     return K
#

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

# Set the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # CCA

# def cca_decomp(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None):
#     """Computes CCA vectors, correlations, and transformed matrices with GPU support."""
#
#     A = torch.tensor(A, dtype=torch.float32).to(device)
#     B = torch.tensor(A, dtype=torch.float32).to(device)
#
#     # Eigen decomposition for A and B
#     if evals_a is None or evecs_a is None:
#         evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
#     if evals_b is None or evecs_b is None:
#         evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
#
#     # Clip small negative eigenvalues to zero
#     evals_a = (evals_a + torch.abs(evals_a)) / 2
#     inv_a = torch.sqrt(torch.where(evals_a > 0, 1 / evals_a, torch.tensor(0.0, device=device)))
#
#     evals_b = (evals_b + torch.abs(evals_b)) / 2
#     inv_b = torch.sqrt(torch.where(evals_b > 0, 1 / evals_b, torch.tensor(0.0, device=device)))
#
#     # Covariance matrix computation
#     cov_ab = A @ B.T
#
#     # Inner SVD problem
#     temp = (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ cov_ab @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T)
#     try:
#         u, s, vh = torch.linalg.svd(temp)
#     except:
#         u, s, vh = torch.linalg.svd(temp * 100)
#         s = s / 100
#
#     del cov_ab
#     del temp
#
#     transformed_a = (u.T @ (evecs_a @ torch.diag(inv_a) @ evecs_a.T) @ A).T
#     transformed_b = (vh @ (evecs_b @ torch.diag(inv_b) @ evecs_b.T) @ B).T
#
#     del A
#     del B
#
#     return u.cpu.numpy(), s.cpu.numpy(), vh.cpu.numpy(), transformed_a.cpu.numpy(), transformed_b.cpu.numpy()

def cca_decomp(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None):
    """Computes CCA vectors, correlations, and transformed matrices
    requires a < n and b < n
    Args:
        A: np.array of size a x n where a is the number of neurons and n is the dataset size
        B: np.array of size b x n where b is the number of neurons and n is the dataset size
    Returns:
        u: left singular vectors for the inner SVD problem
        s: canonical correlation coefficients
        vh: right singular vectors for the inner SVD problem
        transformed_a: canonical vectors for matrix A, a x n array
        transformed_b: canonical vectors for matrix B, b x n array
    """
    # assert A.shape[0] <= A.shape[1]
    # assert B.shape[0] <= B.shape[1]

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / 2
    inv_a = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / 2
    inv_b = np.array([1 / np.sqrt(x) if x > 0 else 0 for x in evals_b])

    cov_ab = A @ B.T

    temp = (
            (evecs_a @ np.diag(inv_a) @ evecs_a.T)
            @ cov_ab
            @ (evecs_b @ np.diag(inv_b) @ evecs_b.T)
    )

    try:
        u, s, vh = np.linalg.svd(temp)
    except:
        u, s, vh = np.linalg.svd(temp * 100)
        s = s / 100

    transformed_a = (u.T @ (evecs_a @ np.diag(inv_a) @ evecs_a.T) @ A).T
    transformed_b = (vh @ (evecs_b @ np.diag(inv_b) @ evecs_b.T) @ B).T
    return u, s, vh, transformed_a, transformed_b


# def mean_sq_cca_corr(rho):
#     """Compute mean squared CCA correlation."""
#     return 1 - torch.sum(rho * rho).item() / len(rho)

def mean_sq_cca_corr(rho):
    """Compute mean squared CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho * rho) / len(rho)


# def mean_cca_corr(rho):
#     """Compute mean CCA correlation."""
#     return 1 - torch.sum(rho).item() / len(rho)

def mean_cca_corr(rho):
    """Compute mean CCA correlation
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    """
    # len(rho) is min(A.shape[0], B.shape[0])
    return 1 - np.sum(rho) / len(rho)


# def pwcca_dist(A, rho, transformed_a):
#     """Computes projection weighted CCA distance between A and B."""
#
#     in_prod = transformed_a.T @ A.T
#     weights = torch.sum(torch.abs(in_prod), axis=1)
#     weights = weights / torch.sum(weights)
#     dim = min(len(weights), len(rho))
#
#     del in_prod
#
#     return 1 - torch.dot(weights[:dim], rho[:dim]).item()

def pwcca_dist(A, rho, transformed_a):
    """Computes projection weighted CCA distance between A and B given the correlation
    coefficients rho and the transformed matrices after running CCA
    :param A: np.array of size a x n where a is the number of neurons and n is the dataset size
    :param B: np.array of size b x n where b is the number of neurons and n is the dataset size
    :param rho: canonical correlation coefficients returned by cca_decomp(A,B)
    :param transformed_a: canonical vectors for A returned by cca_decomp(A,B)
    :param transformed_b: canonical vectors for B returned by cca_decomp(A,B)
    :return: PWCCA distance
    """
    in_prod = transformed_a.T @ A.T
    weights = np.sum(np.abs(in_prod), axis=1)
    weights = weights / np.sum(weights)
    dim = min(len(weights), len(rho))
    return 1 - np.dot(weights[:dim], rho[:dim])


# # CKA

# def lin_cka_dist(A, B):
#     """Computes Linear CKA distance between representations A and B."""
#
#     similarity = torch.norm(B @ A.T, p='fro') ** 2
#     normalization = torch.norm(A @ A.T, p='fro') * torch.norm(B @ B.T, p='fro')
#
#     return (1 - similarity / normalization).item()

def lin_cka_dist(A, B):
    """
    Computes Linear CKA distance bewteen representations A and B
    """
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(
        B @ B.T, ord="fro"
    )
    return 1 - similarity / normalization


# def lin_cka_prime_dist(A, B):
#     """Computes Linear CKA prime distance between representations A and B."""
#
#     if A.shape[0] > A.shape[1]:
#         At_A = A.T @ A  # O(n * n * a)
#         Bt_B = B.T @ B  # O(n * n * a)
#         numerator = torch.sum((At_A - Bt_B) ** 2)
#         denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
#
#         del At_A
#         del Bt_B
#
#         return (numerator / denominator).item()
#     else:
#         similarity = torch.norm(B @ A.T, p='fro') ** 2
#         denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
#         return 1 - 2 * similarity / denominator

def lin_cka_prime_dist(A, B):
    """
    Computes Linear CKA prime distance bewteen representations A and B
    The version here is suited to a, b >> n
    """
    if A.shape[0] > A.shape[1]:
        At_A = A.T @ A  # O(n * n * a)
        Bt_B = B.T @ B  # O(n * n * a)
        numerator = np.sum((At_A - Bt_B) ** 2)
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return numerator / denominator
    else:
        similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
        denominator = np.sum(A ** 2) ** 2 + np.sum(B ** 2) ** 2
        return 1 - 2 * similarity / denominator


# # CKA distance using Gaussian or Laplace kernels

def cka_dist(A, B, A_kernelized=None, B_kernelized=None, kerneltype=1, sigma=100):
    """Computes CKA distance between representations A and B using Gaussian or Laplace kernels."""
    if A_kernelized is None or B_kernelized is None:
        if kerneltype == 1:
            A_kernelized = cuda_rbf_kernel(A.T, sigma=sigma).to(device)
            B_kernelized = cuda_rbf_kernel(B.T, sigma=sigma).to(device)
        else:
            A_kernelized = cuda_laplacian_kernel(A.T, sigma=sigma).to(device)
            B_kernelized = cuda_laplacian_kernel(B.T, sigma=sigma).to(device)

    similarity = torch.norm(B_kernelized @ A_kernelized.T, p='fro') ** 2
    normalization = torch.norm(A_kernelized @ A_kernelized.T, p='fro') * torch.norm(B_kernelized @ B_kernelized.T,
                                                                                    p='fro')

    return (1 - similarity / normalization).item()


# # Procrustes

# def procrustes(A, B):
#     """Computes Procrustes distance between representations A and B."""
#
#     n = A.shape[1]
#     A_sq_frob = torch.sum(A ** 2) / n
#     B_sq_frob = torch.sum(B ** 2) / n
#     nuc = torch.linalg.norm(A @ B.T, ord='nuc') / n  # O(p * p * n)
#
#     return (A_sq_frob + B_sq_frob - 2 * nuc).item()

def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    """
    n = A.shape[1]
    A_sq_frob = np.sum(A ** 2) / n
    B_sq_frob = np.sum(B ** 2) / n
    nuc = np.linalg.norm(A @ B.T, ord="nuc") / n  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc


# # Our predictor distances

# def GULP_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
#     """Computes distance between best linear predictors on representations A and B."""
#
#     n = A.shape[1]
#
#     # Compute eigenvalues and eigenvectors if not provided
#     if evals_a is None or evecs_a is None:
#         evals_a, evecs_a = torch.linalg.eigh(A @ A.T)
#     if evals_b is None or evecs_b is None:
#         evals_b, evecs_b = torch.linalg.eigh(B @ B.T)
#
#     if evals_a is None:
#         inv_a_lambda = None
#     else:
#         evals_a = (evals_a + torch.abs(evals_a)) / (2 * n)
#         inv_a_lmbda = torch.where(evals_a > 0, 1 / (evals_a + lmbda), 1 / lmbda)
#
#     if evals_b is None:
#         inv_b_lambda = None
#     else:
#         evals_b = (evals_b + torch.abs(evals_b)) / (2 * n)
#         inv_b_lmbda = torch.where(evals_b > 0, 1 / (evals_b + lmbda), 1 / lmbda)
#
#     if evals_a is None or evecs_a is None:
#         return None
#     else:
#         T1 = torch.sum(torch.square(evals_a * inv_a_lmbda))
#         T2 = torch.sum(torch.square(evals_b * inv_b_lmbda))
#
#         cov_ab = A @ B.T / n
#         T3 = torch.trace(
#             cov_ab.T @ (evecs_a @ torch.diag(inv_a_lmbda) @ evecs_a.T)
#             @ cov_ab
#             @ (evecs_b @ torch.diag(inv_b_lmbda) @ evecs_b.T)
#         )
#
#         del cov_ab
#         del inv_a_lmbda
#         del inv_b_lmbda
#
#         return (T1 + T2 - 2 * T3).item()

def GULP_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    n = A.shape[1]

    # assert k <= n
    # assert l <= n

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a])
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b])
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])

    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))

    cov_ab = A @ B.T / n
    # T3 = np.trace(
    #     (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
    #     @ cov_ab
    #     @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
    #     @ cov_ab.T
    #     @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    # )

    T3 = np.trace(
        cov_ab.T @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T))

    return T1 + T2 - 2 * T3


# # UKP distance

def UKP_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, A_kernelized=None, B_kernelized=None,
             kerneltype=1, sigma=1, lmbda=0.01):
    """
    Computes distance between best kernel ridge regression predictors on representations A and B, based on Gaussian or Laplace kernels.

    Parameters:
        A (matrix): k by n matrix of representations of first model.
        B (matrix): l by n matrix of representations of second model.
        kerneltype (int): type of kernel to use (1 if Gaussian RBF, 2 if Laplace).
        sigma (float): bandwidth parameter of kernel (same as in rbf_kernel or laplacian_kernel functions in scikit-learn).
        lmbda (float): regularization parameter.

    Returns:
        UKP distance between A and B (scalar).
    """

    n = A.shape[1]

    if A_kernelized is None or B_kernelized is None:
        if kerneltype == 1:
            A_kernelized = cuda_rbf_kernel(A.T, sigma=sigma).to(device)
            B_kernelized = cuda_rbf_kernel(B.T, sigma=sigma).to(device)
        else:
            A_kernelized = cuda_laplacian_kernel(A.T, sigma=sigma).to(device)
            B_kernelized = cuda_laplacian_kernel(B.T, sigma=sigma).to(device)

    if evals_a is None or evecs_a is None:
        try:
            evals_a, evecs_a = torch.linalg.eigh(A_kernelized)
        except torch._C._LinAlgError as e:
            evals_a = None
            evecs_a = None
    if evals_b is None or evecs_b is None:
        try:
            evals_b, evecs_b = torch.linalg.eigh(B_kernelized)
        except torch._C._LinAlgError as e:
            evals_b = None
            evecs_b = None

    if evals_a is None:
        inv_a_lambda = None
    else:
        evals_a = (evals_a + torch.abs(evals_a)) / 2
        inv_a_lmbda = torch.where(evals_a > 0, 1 / (evals_a + n * lmbda), 1 / n * lmbda)

    if evals_b is None:
        inv_b_lambda = None
    else:
        evals_b = (evals_b + torch.abs(evals_b)) / 2
        inv_b_lmbda = torch.where(evals_b > 0, 1 / (evals_b + n * lmbda), 1 / n * lmbda)

    if evals_a is None or evecs_a is None or evals_b is None or evecs_b is None:
        return None
    else:
        T1 = torch.sum(torch.square(evals_a * inv_a_lmbda))
        T2 = torch.sum(torch.square(evals_b * inv_b_lmbda))

        T3 = torch.trace(A_kernelized @ (evecs_a @ torch.diag(inv_a_lmbda) @ evecs_a.T)
                         @ B_kernelized
                         @ (evecs_b @ torch.diag(inv_b_lmbda) @ evecs_b.T))

        del inv_a_lmbda
        del inv_b_lmbda

        return (T1 + T2 - 2 * T3).item()


# def squared_procrustes(A, B):
#     """
#     Computes distance between best linear predictors on representations A and B.
#     """
#
#     n = A.shape[1]
#
#     cov_a = A @ A.T / (n - 1)
#     cov_b = B @ B.T / (n - 1)
#     cov_ab = A @ B.T / (n - 1)
#
#     T1 = torch.trace(cov_a @ cov_a)
#     T2 = torch.trace(cov_b @ cov_b)
#     T3 = torch.trace(cov_ab @ cov_ab.T)
#
#     del cov_a
#     del cov_b
#     del cov_ab
#
#     return (T1 + T2 - 2 * T3).item()

def squared_procrustes(A, B):
    """
    Computes distance bewteen best linear predictors on representations A and B
    """
    n = A.shape[1]

    # assert k < n
    # assert l < n

    cov_a = A @ A.T / (n - 1)
    cov_b = B @ B.T / (n - 1)
    cov_ab = A @ B.T / (n - 1)

    T1 = np.trace(cov_a @ cov_a)
    T2 = np.trace(cov_b @ cov_b)
    T3 = np.trace(cov_ab @ cov_ab.T)

    return T1 + T2 - 2 * T3


# # Defining kernel functions

def cuda_rbf_kernel(X, Y=None, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between two sets of samples.

    Parameters:
        X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Second input tensor of shape (n_samples_Y, n_features).
                        If None, the kernel is computed with X itself.
        sigma (float): Bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: RBF kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    if Y is None:
        Y = X

    # Ensure both inputs are on the same device (CUDA)
    if not (X.is_cuda and Y.is_cuda):
        raise ValueError("Both inputs must be CUDA tensors.")

    # Compute squared Euclidean distance
    X_sq = torch.sum(X ** 2, dim=1, keepdim=True)  # shape (n_samples_X, 1)
    if Y is None:
        Y_sq = X_sq
    else:
        Y_sq = torch.sum(Y ** 2, dim=1, keepdim=True)  # shape (n_samples_Y, 1)

    # Compute the squared distance matrix
    dists = X_sq + Y_sq.t() - 2 * torch.mm(X, Y.t())  # shape (n_samples_X, n_samples_Y)

    # Apply the RBF kernel formula
    K = torch.exp(-dists / (2 * sigma ** 2))

    del dists
    del X_sq
    del Y_sq

    return K


def cuda_laplacian_kernel(X, Y=None, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between two sets of samples.

    Parameters:
        X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Second input tensor of shape (n_samples_Y, n_features).
                        If None, the kernel is computed with X itself.
        sigma (float): Bandwidth parameter for the RBF kernel.

    Returns:
        torch.Tensor: RBF kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    if Y is None:
        Y = X

    # Ensure both inputs are on the same device (CUDA)
    if not (X.is_cuda and Y.is_cuda):
        raise ValueError("Both inputs must be CUDA tensors.")

    # Compute the L1 distance matrix
    dists = torch.cdist(X, Y, p=1)

    # Apply the RBF kernel formula
    K = torch.exp(-dists / sigma)

    del dists

    return K

