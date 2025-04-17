import numpy as np
from scipy.stats import norm, zscore
from sklearn.mixture import GaussianMixture
import scipy.sparse
from heapq import nlargest
from ADMM_spare_ortho_dic_encode import ADMM_spare_ortho_dic_encode

def scan_samp(X, H, HatomInfo, Phi, threshold, rho, lambdaa, beta):

	t, f = X.shape

	# Draw random samples samples for stopping criteria
	size_ = 3
	size_dist = 200000
	r_end = t - size_
	c_end = f - size_
	distribution = np.zeros(size_dist)

	#Generate random indices for the top-left corner of the 5x5 matrices
	rowIndices = np.random.randint(r_end, size=size_dist)
	colIndices = np.random.randint(c_end, size=size_dist)

	for i in range(size_dist):
		subMatrice = X[rowIndices[i]:rowIndices[i] + size_, colIndices[i]:colIndices[i] + size_]
		distribution[i] = np.mean(subMatrice)

	pd_mu, pd_std = norm.fit(distribution)

	# intilization
	Omega = np.ones(X.shape)
	p, f = H.shape

	sum_H = H.sum(axis=1)
	flat_X = np.sum(X, 0)

	# subselecting H atom to speed up SCAN
	a = (scipy.sparse.csc_array(np.array([flat_X])).dot(H.T.tocsc()) / sum_H).toarray()[0]
	temp = nlargest(int(p * beta + .5), np.ndenumerate(a), key=lambda x:x[1])
	potFreqInd = [x[0][0] for x in temp] # x[0][0] because x is ((idx,), a[idx])
	H = H.tocsr()[potFreqInd, :]
	HatomInfo = HatomInfo[potFreqInd, :]

	Norm_H = scipy.sparse.linalg.norm(H.T, axis=0)
	# Norm_H is 1xp (length of each freq atom)

	pred = []
	W_c = []
	i = 1
	while True:
		mu_X = np.sum(X * Omega, axis=1) / np.sum(Omega, axis=1)
		Z = (X - np.outer(mu_X, np.ones(X.shape[1]))) * Omega
		A = scipy.sparse.csr_array(Z).dot(H.T).toarray()
		# Z is txf, H is pxf, so A is txp

		cc = A / Norm_H
		a = np.mean(cc, axis=0)
		# a is 1xp
		m = np.argmax(a)
		chosen_atom = H.tocsr()[m,:]
		atom_start = HatomInfo[m,1]
		atom_end = HatomInfo[m,2]
		ident_f = list(range(atom_start, atom_end + 1))

		y = A[:, m]
		# A is txp -> y is tx1
		y = zscore(y)
		# y is a column vector (but not really bc numpy)

		#Phi is txp
		W, _ = ADMM_spare_ortho_dic_encode(y.T,Phi.T,lambdaa,rho);
		W = W.T
		# W is px1
		W_c.append(W)

		PhiW = Phi.dot(W)
		#PhiW is tx1

		# time step identification
		NN=2
		gmm = GaussianMixture(NN, max_iter=1000, covariance_type="diag", reg_covar=0.000001).fit(PhiW.reshape(-1,1))
		cluster_means = gmm.means_
		mean_idx = np.argmax(cluster_means, axis=0)
		cluster_labels = gmm.predict(PhiW.reshape(-1,1))

		# building hole
		ident_t = np.nonzero(np.ravel(cluster_labels == mean_idx, order="F"))[0]
		ident_subs = np.zeros((len(ident_t)*len(ident_f),2))
		offset = 0

		for it in ident_t:
			ident_subs[offset:offset+len(ident_f), 0] = [it] * len(ident_f)
			ident_subs[offset:offset+len(ident_f), 1] = ident_f
			offset = offset + len(ident_f)
		
		trans_ind = [(r, c) for r,c in ident_subs.astype(int)]
		# trans_ind = np.ravel_multi_index((ident_subs[:,0].astype(int),ident_subs[:,1].astype(int)), [t,f], order="F")
		det_box = [X[pt] for pt in trans_ind]
		mean_det = np.mean(det_box)
		p_value = 1-norm.cdf((mean_det - pd_mu) / pd_std)

		print(p_value, threshold, not np.isnan(p_value) and p_value <= threshold)

		if not np.isnan(p_value) and p_value <= threshold:
			pred.append(np.zeros((t,f)))

			for pt in trans_ind:
				Omega[pt] = 0
				pred[-1][pt] = 1

		else:
			break

	if len(pred) == 0:
		pred.append(np.zeros((t,f)))
	return Omega, pred
		
