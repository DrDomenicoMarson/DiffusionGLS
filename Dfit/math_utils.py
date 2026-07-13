
import numpy as np
import numba as nb

@nb.jit(nopython=True)
def setupc(m,n):
    """Build the MSD covariance prefactor matrix from Eq. 8b.

    Parameters
    ----------
    m : int
        Number of MSD lag points used in the fit.
    n : int
        Number of available lag steps in the strided trajectory minus one.

    Returns
    -------
    ndarray of shape (m, m)
        Covariance prefactor matrix (without the ``sigma^4 / 3`` factor).
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
    """Compute the noise correction term for one covariance element.

    Parameters
    ----------
    a2 : float
        Offset parameter of the MSD model.
    s2 : float
        Slope parameter of the MSD model.
    n : int
        Number of lag steps in the strided series minus one.
    i : int
        One-based row index of the covariance element.
    j : int
        One-based column index of the covariance element.

    Returns
    -------
    float
        Additive noise contribution for covariance element ``(i, j)`` from
        Eq. 8a.
    """
    dirac = 1. if i==j else 0.
    add_cov = (
        ((1.+dirac)*a2**2 + 4. * a2 * s2 * min(i,j)) / (n-min(i,j)+1.) 
        + a2**2 * max(n-i-j+1.,0.) / ((n-i+1.) * (n-j+1.))
            )
    return add_cov

@nb.jit(nopython=True)
def calc_cov(n,m,c,a2,s2):
    """Assemble the full covariance matrix used in GLS fitting.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    m : int
        Number of lag points included in the fit.
    c : ndarray of shape (m, m)
        Prefactor matrix returned by :func:`setupc`.
    a2 : float
        Offset parameter of the MSD model.
    s2 : float
        Slope parameter of the MSD model.

    Returns
    -------
    ndarray of shape (m, m)
        Full covariance matrix from Eq. 8a including ``s2**2/3`` scaling.
    """
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
    """Invert a covariance matrix with diagonal regularization.

    Parameters
    ----------
    A : ndarray of shape (m, m)
        Matrix to invert.

    Returns
    -------
    ndarray of shape (m, m)
        Inverse of a regularized copy of ``A``.

    Notes
    -----
    A small diagonal value (``1e-6``) is added to improve numerical
    robustness for near-singular matrices.
    """
    # Regularization to prevent singularity.
    # Copy to avoid mutating the caller's matrix.
    B = A.copy()
    n = B.shape[0]
    for i in range(n):
        B[i,i] += 1e-6
    return np.linalg.inv(B)

@nb.jit(nopython=True)
def calc_a2s2(m,v,cinv, a2, s2):
    """Perform one GLS update step for ``a2`` and ``s2``.

    Parameters
    ----------
    m : int
        Number of MSD lag points used in the fit.
    v : ndarray of shape (m,)
        MSD values at lags ``1..m``.
    cinv : ndarray of shape (m, m)
        Inverse covariance matrix for current GLS iteration.
    a2 : float
        Current offset estimate.
    s2 : float
        Current slope estimate.

    Returns
    -------
    tuple[float, float, float]
        ``(a2_new, s2_new, d2)`` where ``d2`` is the squared update distance
        used by the convergence criterion.
    """
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

@nb.jit(nopython=True)
def gls_closed(n,v, c=None):
    """Compute a closed-form initialization for GLS parameters.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    v : ndarray of shape (m,)
        MSD values at lags ``1..m``. The first two values are required.
    c : ndarray or None, optional
        Optional precomputed covariance prefactor for ``m=2``.

    Returns
    -------
    tuple[float, float, float]
        ``(a2, s2, s2var)`` estimated from the ``m=2`` closed-form formulas.
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

@nb.jit(nopython=True)
def gls_iter(n,m,v,a2,s2,d2max,nitmax, c=None):
    """Iteratively refine GLS parameters for the MSD linear model.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    m : int
        Number of lag points used in the fit.
    v : ndarray of shape (m,)
        MSD values at lags ``1..m``.
    a2 : float
        Initial offset estimate.
    s2 : float
        Initial slope estimate.
    d2max : float
        Convergence threshold for squared parameter updates.
    nitmax : int
        Maximum number of iterations.
    c : ndarray or None, optional
        Optional precomputed covariance prefactor for ``m``.

    Returns
    -------
    tuple[float, float, int]
        ``(a2, s2, nit)`` containing the final parameters and iteration count.
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

@nb.jit(nopython=True)
def calc_gls(n,m,v,d2max,nitmax, c2=None, cm=None):
    """Estimate GLS parameters with iterative refinement and safe fallback.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    m : int
        Number of lag points used in the fit.
    v : ndarray of shape (m,)
        MSD values at lags ``1..m``.
    d2max : float
        Convergence threshold for iterative updates.
    nitmax : int
        Maximum number of iterations.
    c2 : ndarray or None, optional
        Optional precomputed prefactor matrix for the ``m=2`` closed-form
        initialization.
    cm : ndarray or None, optional
        Optional precomputed prefactor matrix for iterative updates with ``m``.

    Returns
    -------
    tuple[float, float, bool]
        ``(a2, s2, converged)``. If convergence is not reached within
        ``nitmax``, the function falls back to the closed-form ``m=2`` result
        and returns ``converged=False``.
    """
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
    """Evaluate chi-squared for the 3D MSD fit.

    Parameters
    ----------
    m : int
        Number of lag points included in the fit.
    a2_3D : float
        3D offset parameter (sum across dimensions).
    s2_3D : float
        3D slope parameter (sum across dimensions).
    msds_3D : ndarray of shape (m,)
        3D MSD values.
    cinv : ndarray of shape (m, m)
        Inverse covariance matrix.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    float
        Chi-squared value multiplied by ``ndim``.
    """
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
    """Estimate the variance of ``s2`` for one dimension.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    m : int
        Number of lag points used in the fit.
    a2 : float
        Offset estimate for one dimension.
    s2 : float
        Slope estimate for one dimension.

    Returns
    -------
    float
        Estimated variance of ``s2`` for the given dimension.
    """

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

@nb.jit(nopython=True)
def eval_vars(n,m,a2m,s2m,ndim):
    """Aggregate slope variance estimates across dimensions.

    Parameters
    ----------
    n : int
        Number of lag steps in the strided series minus one.
    m : int
        Number of lag points used in the fit.
    a2m : ndarray of shape (ndim,)
        Mean offset estimates across segments for each dimension.
    s2m : ndarray of shape (ndim,)
        Mean slope estimates across segments for each dimension.
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    float
        Sum of per-dimension ``s2`` variance estimates.
    """
    s2var = 0.0
    for d in range(ndim):
        s2var += calc_var(n,m,a2m[d],s2m[d]) # using mean across segments but per dim
    return s2var

def compute_autocorrelation_via_fft(x):
    """Compute a normalized autocorrelation function via FFT.

    Parameters
    ----------
    x : ndarray of shape (n,)
        One-dimensional signal values.

    Returns
    -------
    ndarray of shape (n,)
        Autocorrelation values normalized by the number of overlapping samples
        at each lag.
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
    """Compute one-dimensional MSD using FFT-based autocorrelation.

    Parameters
    ----------
    x : ndarray of shape (n,)
        One-dimensional coordinate trajectory.

    Returns
    -------
    ndarray of shape (n,)
        MSD values for lags ``0..n-1``.

    Notes
    -----
    The implementation uses cumulative sums and FFT correlation to avoid
    quadratic-time Python loops.
    """
    corrx = compute_autocorrelation_via_fft(x)
    nt = len(x)
    dsq = x**2

    # Vectorized cumulative subtraction from both ends
    cumsum_front = np.cumsum(dsq)        # cumsum_front[m-1] = sum(dsq[0:m])
    cumsum_back = np.cumsum(dsq[::-1])   # cumsum_back[m-1] = sum of last m elements

    ms = np.arange(1, nt)
    sumsq_arr = 2 * np.sum(dsq) - cumsum_front[ms - 1] - cumsum_back[ms - 1]

    msd = np.zeros(nt)
    msd[1:] = sumsq_arr / (nt - ms) - 2 * corrx[1:]
    return msd

@nb.jit(nopython=True)
def compute_MSD_1D_first_m(x, m):
    """Compute MSD values for lags ``1..m`` using direct summation.

    This is an O(n × m) computation that only calculates the first ``m``
    lag values needed for GLS fitting.  Because it is fully Numba-compiled
    in nopython mode, it releases the GIL and allows true multithreading.

    Parameters
    ----------
    x : ndarray of shape (n,)
        One-dimensional coordinate trajectory.
    m : int
        Number of MSD lag values to compute.

    Returns
    -------
    ndarray of shape (m,)
        MSD values at lags ``1..m``.
    """
    n = len(x)
    msd = np.zeros(m)
    for lag in range(1, m + 1):
        total = 0.0
        count = n - lag
        for i in range(count):
            diff = x[i + lag] - x[i]
            total += diff * diff
        msd[lag - 1] = total / count
    return msd
