import numpy as np
def ADMM_spare_ortho_dic_encode(x, phi, lambdaa, rho):
    # x is 1xt, lambdaa and rho are scalars
    p, t = phi.shape
    num_iters = 500
    tol = 1e-6

    I = np.eye(p)
    w = np.random.rand(p)
    z = w
    Gamma = 0
    #1xp

    fit = 0
    #1xt
    for i in range(num_iters):
        # update w eq 12
        w = (x.dot(phi.T)+rho*z+Gamma) / (1+rho)

        #update z eq 14, 15
        h = w-Gamma/rho
        z = np.sign(h) * max(np.abs(h) - lambdaa/rho,)

        # update Gamma eq 16
        Gamma = Gamma + rho*(z-w)

        old_fit = fit
        fit = np.linalg.norm(x-w*phi) + np.sum(np.abs(z))

        fit_change = fit - old_fit

        #if our fit hasnt changed much return

        if np.abs(fit_change) < tol:
            return w, z

    return w,z

