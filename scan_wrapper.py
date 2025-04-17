import numpy as np
import scipy.io
import scipy.sparse
import h5py
from pathlib import Path
from H_sparse_gen import H_sparse_gen
from generate_haar import generate_haar
from scan_samp import scan_samp

def _decompose_pre_computed_H(pre_computed_H, f=None):

	data = pre_computed_H["H"]["data"][:].astype(int)
	ir = pre_computed_H["H"]["ir"][:]
	jc = pre_computed_H["H"]["jc"][:]
	atomInfo = pre_computed_H["atomInfo"][:].astype(int)

	# this part is designed to handle .mat files of sparse matricies created via matlab
	# if len(jc) == f:
		# jc_extended = np.empty(len(ir), dtype=int)
		# for i, j in enumerate(jc[:-1]):
		# 	nextj = jc[i+1]
		# 	jc_extended[j:nextj] = i
		# jc = jc_extended
		# atomInfo = atomInfo.T

	H_all = scipy.sparse.coo_array((data, (ir, jc)))

	return H_all, atomInfo

def scan_wrapper(data):
	data = np.array(data)
	t1, f = data.shape
	t = 2**int(np.log2(t1))
	data = data[:t, :]

	prefix = Path("precomputedH")

	if (prefix / f"H_{f}b.mat").exists():
		with h5py.File(prefix / f"H_{f}b.mat","r") as pre_computed_H:
			H_all, HatomInfo = _decompose_pre_computed_H(pre_computed_H)

	else:
		H, atomInfo = H_sparse_gen(f)
		H_var = f"H_{f}b.mat"

		ir, jc = H.coords
		
		with h5py.File(prefix / H_var, "w") as f:
			f.create_group("H")
			f["H"]["ir"] = ir
			f["H"]["jc"] = jc
			f["H"]["data"] = H.data
			f["atomInfo"] = atomInfo

		H_all = H
		HatomInfo = atomInfo

	lambdaa = 0.0000001
	rho = 10*lambdaa
	threshold=0.15
	beta = 1
	Phi = generate_haar(t)
	Phi = Phi.T
	#Phi is txp

	return scan_samp(data, H_all, HatomInfo, Phi, threshold, rho, lambdaa, beta)
