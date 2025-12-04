
import numpy as np
import numba as nb

@nb.jit(nopython=True)
def setupc(m,n):
    """
    Setup covariance matrix for MSD(k)=<(z(i+k)-z(i))^2> 
    Eq. 8b expression in [ brackets ] (excluding sig^4/3)
    """
    c = np.zeros((m,m))

    for i in range(1,m+1):
        for j in range(1,m+1):

            # Heaviside
            if i + j - n - 2 >= 0:
                heav = 1.0
            else:
                heav = 0.0

            c[i-1,j-1] = (
                2.*min(i,j) * (1.+3.*i*j - min(i,j)**2) / (n - min(i,j) +1) 
                + (min(i,j)**2 - min(i,j)**4) / ((n-i+1.)*(n-j+1.))
                + heav * ((n+1.-i-j)**4 - (n+1.-i-j)**2) / ((n-i+1.)*(n-j+1.))
                )
    return c

@nb.jit(nopython=True)
def add_to_cov(a2,s2,n,i,j):
    """ Noise contribution Eq. 8a to covariance matrix element ij. """
    dirac = 1. if i==j else 0.
    add_cov = (
        ((1.+dirac)*a2**2 + 4. * a2 * s2 * min(i,j)) / (n-min(i,j)+1.) 
        + a2**2 * max(n-i-j+1.,0.) / ((n-i+1.) * (n-j+1.))
            )
    return add_cov

@nb.jit(nopython=True)
def calc_cov(n,m,c,a2,s2):
    """ Eq. 8a, with factor s^4/3 from Eq. 8b """
    cov = np.zeros((m,m))
    for i in range(1,m):
        for j in range(i+1,m+1):
            cov[i-1,j-1] = c[i-1,j-1] * s2**2 / 3. + add_to_cov(a2,s2,n,i,j)

    cov += cov.T
    for i in range(1,m+1):
        cov[i-1,i-1] = c[i-1,i-1] * s2**2 / 3. + add_to_cov(a2,s2,n,i,i)
    return cov

@nb.jit(nopython=True)
def inv_mat(A):
    # Regularization to prevent singularity
    # We need to copy A to avoid modifying the input if it's used elsewhere
    # But A is usually created in calc_cov.
    # Let's add epsilon to diagonal.
    n = A.shape[0]
    for i in range(n):
        A[i,i] += 1e-6
    cinv = np.linalg.inv(A)
    return cinv

@nb.jit(nopython=True)
def calc_a2s2(m,v,cinv, a2, s2):
    A = 0.0
    B = 0.0
    C = 0.0
    D = 0.0
    E = 0.0

    for i in range(m):
        for j in range(m):
            A += cinv[i,j]
            B += (i+1) * cinv[i,j]
            C += (i+1) * (j+1) * cinv[i,j]
            D += v[i] * cinv[i,j]
            E += (i+1) * v[j] * cinv[i,j]

    denom = A * C - B**2

    a2n = (C*D-B*E) / denom
    s2n = (A*E-B*D) / denom

    d2=(s2n-s2)**2+(a2n-a2)**2
    return a2n, s2n, d2

def gls_closed(n,v, c=None):
    """
    Closed-form GLS estimation of offset a^2 and variance sigma^2.
    """

    # c = setupc(2,n)
    if c is None:
        c = setupc(2,n) # setups of cov matrix

    s2 = v[1]-v[0]
    a2 = 2*v[0] - v[1]

    cov = calc_cov(n,2,c,a2,s2)
    #_cinv = inv_mat(cov)

    # a2var = abs(4. * cov[0,0] - 4. * cov[0,1] + cov[1,1])
    s2var = abs(cov[0,0] - 2. * cov[0,1] + cov[1,1])

    return a2, s2, s2var

def gls_iter(n,m,v,a2,s2,d2max,nitmax, c=None):
    """
    Iteratively optimize offset a^2 and variance sigma^2
    of fit to MSD for Gaussian random walk with noise
    (model: MSD(i)=a^2+i*sigma^2).
    """

    nit = 0
    d2 = d2max + 1.e5

    if c is None:
        c = setupc(m,n) # setups of cov matrix

    while (nit < nitmax) and (d2 > d2max):
        nit = nit+1
        cov = calc_cov(n,m,c,a2,s2)

        # calculate inverse covariance
        cinv = inv_mat(cov)

        # estimate new values for a^2 and sigma^2
        a2, s2, d2 = calc_a2s2(m, v, cinv, a2, s2)

    return a2, s2, nit

def calc_gls(n,m,v,d2max,nitmax, c2=None, cm=None):
    a2est, s2est, _s2varest = gls_closed(n,v, c=c2)
    a2, s2, nit = gls_iter(n,m,v,a2est,s2est,d2max,nitmax, c=cm)

    converged = True
    if nit >= nitmax:
        # print('WARNING: Optimizer did not converge. Falling back to M=2.')
        a2, s2 = a2est, s2est # use M=2 result
        converged = False

    return a2, s2, converged

@nb.jit(nopython=True)
def calc_chi2(m,a2_3D,s2_3D,msds_3D,cinv,ndim):
    chi2 = 0.0
    for i in range(m):
        for j in range(m):
            chi2 += (msds_3D[i] - a2_3D - s2_3D * (i+1.)) * (
                cinv[i,j] * (msds_3D[j] - a2_3D - s2_3D * (j+1.))
                )
    chi2 = chi2*ndim
    return chi2

@nb.jit(nopython=True)
def calc_var(n,m,a2,s2):

    c = setupc(m,n)
    cov = calc_cov(n,m,c,a2,s2)
    cinv = inv_mat(cov)

    A = 0.0
    B = 0.0
    C = 0.0

    for i in range(m):
        for j in range(m):
            A += cinv[i,j]
            B += (i+1) * cinv[i,j]
            C += (i+1) * (j+1) * cinv[i,j]
    denom = A * C - B**2
    # a2var = C / denom
    s2var = A / denom
    return s2var

def eval_vars(n,m,a2m,s2m,ndim):
    """ a2 and s2 are means across segments, 
    but still per dim """
    s2var = 0.0
    for d in range(ndim):
        s2var += calc_var(n,m,a2m[d],s2m[d]) # using mean across segments but per dim
    return s2var

def compute_autocorrelation_via_fft(x):
    """
    Autocorrelation of array calculated via FFT.
    """
    # x is assumed to be a numpy array (float64)
    l = len(x)
    
    # FFT
    xft = np.fft.fft(x, 2*l)
    
    # Autocorrelation: ifft( |xft|^2 )
    # np.abs(xft)**2 is real, but ifft expects complex? 
    # Actually correlation is ifft( conj(xft) * xft ) = ifft( |xft|^2 )
    # |xft|^2 is real.
    
    # Numba supports basic arithmetic on complex arrays
    spec = np.abs(xft)**2
    corr = np.real(np.fft.ifft(spec))
    
    # Normalization
    norm = l - np.arange(l)
    # Avoid division by zero if l=0 (unlikely)
    # Numba handles array division
    
    # Slice to length l
    corr = corr[:l]/norm
    return corr

def compute_MSD_1D_via_correlation(x):
    """
    One-dimensional MSD calculated via FFT-based auto-correlation.
    """
    corrx = compute_autocorrelation_via_fft(x)
    nt     = len(x)
    dsq    = x**2
    sumsq  = 2*np.sum(dsq)
    msd    = np.zeros((nt))
    msd[0] = 0.
    for m in range(1,nt):
        sumsq  = sumsq - dsq[m-1]-dsq[nt-m]
        msd[m] = sumsq/(nt-m) - 2*corrx[m]
    return msd
