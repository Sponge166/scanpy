import scipy.sparse
import numpy as np
def H_sparse_gen(t):
	atomInfo = []
	iInds = []
	jInds = []
	atomInd = 0

	for nnzCol in range(t):
		tpoint = 0
		for i in range(t-nnzCol):
			AtomStart = tpoint
			AtomEnd = tpoint+nnzCol
			tpoint += 1

			atomInfo.append([atomInd, AtomStart, AtomEnd])
			jInds.extend(list(range(AtomStart, AtomEnd + 1)))
			rowInds = [atomInd] * (nnzCol + 1)
			iInds.extend(rowInds)

			atomInd += 1

	oneVec = [1] * len(iInds)

	S = scipy.sparse.coo_array((oneVec, (iInds, jInds)))

	return S, atomInfo




