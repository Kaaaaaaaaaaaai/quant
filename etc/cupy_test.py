import numpy as np
import cupy as cp
import time

# def legacy_corr(A:np.ndarray) -> np.ndarray:

#     rho = np.array([[np.corrcoef(A[i, ::], A[j, ::])[0, 1] for j in range(A.shape[0])] for i in range(A.shape[0])])

#     return rho

def corr_batch_np(A:np.ndarray) -> np.ndarray:

    return np.dot(A, A.T) / np.sqrt(np.dot(np.dot((np.dot(A, A.T) * np.eye(A.shape[0])), np.ones((A.shape[0], A.shape[0]))),(np.dot(A, A.T) * np.eye(A.shape[0]))))

def corr_batch_cp(A:cp.ndarray) -> cp.ndarray:

    assert cp.cuda.is_available(), "CuPy is not configured to use CUDA. Please check your CuPy installation."

    return cp.divide(cp.dot(A, A.T), cp.sqrt(cp.dot(cp.dot((cp.dot(A, A.T) * cp.eye(A.shape[0])), cp.ones((A.shape[0], A.shape[0]))), (cp.dot(A, A.T) * cp.eye(A.shape[0])))))

A = np.random.rand(10000, 120)

# start = time.time()
# legacy_result = legacy_corr(A)
# print(f"Legacy correlation took {time.time() - start:.2f} seconds")

start = time.time()
batch_result_np = corr_batch_np(A - np.mean(A, axis=0, keepdims=True))
print(f"Batch correlation (NumPy) took {time.time() - start:.2f} seconds")

A = cp.array(A)

start = time.time()
batch_result_cp = corr_batch_cp(A - cp.mean(A, axis=0, keepdims=True))
print(f"Batch correlation (CuPy) took {time.time() - start:.2f} seconds")

# assert np.allclose(legacy_result, batch_result_np), "Legacy and NumPy batch results do not match!"
# assert cp.allclose(cp.asnumpy(batch_result_cp), legacy_result), "Legacy and CuPy batch results do not match!"
assert cp.allclose(batch_result_cp, batch_result_np), "NumPy and CuPy batch results do not match!"