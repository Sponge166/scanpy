import numpy as np
def generate_haar(N):
	Nlog2 = np.log2(N)
	if N < 2 or (Nlog2 - int(Nlog2)) != 0:
		raise ValueError('The input argument should be of form 2^k')

	p = np.array([0, 0], dtype=int)
	q = np.array([0, 1], dtype=int)

	n = int(Nlog2 + 1)

	for i in range(1, n):
		p = np.append(p, np.full(2**i, i))
		q = np.append(q, np.arange(1, 2**i + 1, dtype=int))

	Hr = np.zeros((N, N))
	Hr[0, :] = 1

	for i in range(1, N):
		P = p[i]
		Q = q[i]
		for j in range(int(N*(Q-1)/(2**P)), int(N*((Q-0.5)/(2**P)))):
			Hr[i, j] = 2**(P/2)
		for j in range(int(N*((Q-0.5)/(2**P))), int((N*(Q/(2**P))))):
			Hr[i, j] = -(2**(P/2))

	Hr = Hr * (1/np.sqrt(N))

	return Hr


