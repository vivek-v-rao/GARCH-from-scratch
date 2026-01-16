import numpy as np
import pandas as pd
from math import gamma as gamma_fn, sqrt, pi

def simulate_gjr_garch_t(nobs, omega, alpha, gamma, beta, nu,
                         burn=5000, seed=123, h0=None, floor=1e-14):
    """
    simulate gjr-garch(1,1) with student t innovations standardized to var(z)=1
    eps_t = sqrt(h_t) * z_t
    h_t   = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*1{eps_{t-1}<0} + beta*h_{t-1}
    """
    if nu <= 2:
        raise ValueError("nu must be > 2")
    rng = np.random.default_rng(seed)
    # z ~ t_nu scaled to var 1
    t_scale = sqrt((nu - 2.0) / nu)
    z = rng.standard_t(df=nu, size=int(nobs) + int(burn)) * t_scale
    kappa = alpha + 0.5 * gamma + beta
    if h0 is None:
        if kappa < 1.0:
            h_prev = omega / (1.0 - kappa)
        else:
            h_prev = max(omega, floor)
    else:
        if h0 <= 0:
            raise ValueError("h0 must be > 0")
        h_prev = float(h0)
    n_total = int(nobs) + int(burn)
    h = np.empty(n_total, dtype=float)
    eps = np.empty(n_total, dtype=float)
    for t in range(n_total):
        eps_t = sqrt(h_prev) * z[t]
        h[t] = h_prev
        eps[t] = eps_t
        ind = 1.0 if eps_t < 0.0 else 0.0
        h_next = omega + alpha * (eps_t * eps_t) + gamma * (eps_t * eps_t) * ind + beta * h_prev
        if h_next <= floor:
            h_next = floor
        h_prev = h_next
    return eps[burn:], h[burn:]

def acf(x, nlags):
    """autocorrelation for lags 0..nlags"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    c = np.correlate(x, x, mode="full")[n - 1:n + nlags]
    c0 = c[0]
    return c / c0 if c0 > 0 else np.zeros(nlags + 1)

def crosscorr(x, y, nlags):
    """corr(x_t, y_{t+k}) for k=0..nlags"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    vx = np.mean(x * x)
    vy = np.mean(y * y)
    if vx <= 0 or vy <= 0:
        return np.zeros(nlags + 1)
    out = np.empty(nlags + 1, dtype=float)
    for k in range(nlags + 1):
        out[k] = np.mean(x[:x.size - k] * y[k:]) / sqrt(vx * vy)
    return out

def m4_student_t_standardized(nu):
    """E[z^4] for z = t_nu scaled to var 1 (requires nu>4)"""
    if nu <= 4:
        raise ValueError("need nu>4 for finite fourth moment")
    return 3.0 * (nu - 2.0) / (nu - 4.0)

def abs_moment_student_t_standardized(nu, r):
    """E[|z|^r] for z = t_nu scaled to var 1 (requires r<nu and nu>2)"""
    if nu <= 2:
        raise ValueError("need nu>2")
    if r >= nu:
        raise ValueError("need r < nu")
    # E|z|^r = (nu-2)^(r/2) * Gamma((r+1)/2) * Gamma((nu-r)/2) / (sqrt(pi)*Gamma(nu/2))
    return (nu - 2.0) ** (r / 2.0) * gamma_fn((r + 1.0) / 2.0) * gamma_fn((nu - r) / 2.0) / (
        sqrt(pi) * gamma_fn(nu / 2.0)
    )

def theoretical_acf_sq_returns_gjr(omega, alpha, gamma, beta, nu, nlags):
    """
    theoretical acf of s_t=eps_t^2 for symmetric z with var 1 and finite 4th moment
    rho(k) = rho(1)*kappa^(k-1), k>=1
    """
    kappa = alpha + 0.5 * gamma + beta
    if kappa >= 1.0:
        raise ValueError("need alpha + 0.5*gamma + beta < 1")
    m4 = m4_student_t_standardized(nu)
    a2 = alpha * alpha + alpha * gamma + 0.5 * gamma * gamma
    eta = beta * beta + 2.0 * beta * (alpha + 0.5 * gamma) + a2 * m4
    if eta >= 1.0:
        raise ValueError("need eta < 1 for finite E[h^2] (finite fourth moment of eps)")
    hbar = omega / (1.0 - kappa)
    eh2 = (omega * omega) * (1.0 + kappa) / ((1.0 - kappa) * (1.0 - eta))
    var_s = m4 * eh2 - hbar * hbar
    cov1 = omega * hbar + ((alpha + 0.5 * gamma) * m4 + beta) * eh2 - hbar * hbar
    rho1 = cov1 / var_s
    rho = np.empty(nlags + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, nlags + 1):
        rho[k] = rho1 * (kappa ** (k - 1))
    info = {"kappa": kappa, "m4": m4, "eta": eta, "hbar": hbar, "eh2": eh2, "rho1": rho1}
    return rho, info

def theoretical_crosscorr_return_future_sq(omega, alpha, gamma, beta, nu, nlags, eh32):
    """
    corr(eps_t, eps_{t+k}^2), k>=1
    corr(k) = corr(1)*kappa^(k-1)
    level uses eh32 = E[h^(3/2)] (estimated from a long stationary run)
    """
    kappa = alpha + 0.5 * gamma + beta
    if kappa >= 1.0:
        raise ValueError("need kappa < 1")
    # var(eps) = E[eps^2] = hbar
    hbar = omega / (1.0 - kappa)
    # var(eps^2)
    _, info = theoretical_acf_sq_returns_gjr(omega, alpha, gamma, beta, nu, 1)
    m4 = info["m4"]
    eh2 = info["eh2"]
    var_s = m4 * eh2 - hbar * hbar
    # cov(eps_t, eps_{t+1}^2) = -0.5*gamma*E[h^(3/2)]*E|z|^3
    ez_abs3 = abs_moment_student_t_standardized(nu, 3.0)
    cov1 = -0.5 * gamma * eh32 * ez_abs3
    corr1 = cov1 / sqrt(hbar * var_s)
    out = np.empty(nlags + 1, dtype=float)
    out[0] = 0.0
    for k in range(1, nlags + 1):
        out[k] = corr1 * (kappa ** (k - 1))
    info2 = {"kappa": kappa, "cov1": cov1, "corr1": corr1, "ez_abs3": ez_abs3}
    return out, info2

def simulate_garch_t(nobs, omega, alpha, beta, nu,
                     burn=5000, seed=123, h0=None, floor=1e-14):
    """
    simulate garch(1,1) with student t innovations standardized to var(z)=1
    eps_t = sqrt(h_t) * z_t
    h_t   = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
    """
    if nu <= 2:
        raise ValueError("nu must be > 2")
    rng = np.random.default_rng(seed)
    # z ~ t_nu scaled to var 1
    t_scale = sqrt((nu - 2.0) / nu)
    z = rng.standard_t(df=nu, size=int(nobs) + int(burn)) * t_scale
    kappa = alpha + beta
    if h0 is None:
        if kappa < 1.0:
            h_prev = omega / (1.0 - kappa)
        else:
            h_prev = max(omega, floor)
    else:
        if h0 <= 0:
            raise ValueError("h0 must be > 0")
        h_prev = float(h0)
    n_total = int(nobs) + int(burn)
    h = np.empty(n_total, dtype=float)
    eps = np.empty(n_total, dtype=float)
    for t in range(n_total):
        eps_t = sqrt(h_prev) * z[t]
        h[t] = h_prev
        eps[t] = eps_t
        h_next = omega + alpha * (eps_t * eps_t) + beta * h_prev
        if h_next <= floor:
            h_next = floor
        h_prev = h_next
    return eps[burn:], h[burn:]

def theoretical_acf_sq_returns_garch(omega, alpha, beta, nu, nlags):
    """
    theoretical acf of s_t=eps_t^2 for symmetric garch(1,1) with standardized t innovations
    model:
      h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}
      eps_t = sqrt(h_t) * z_t
      E[z_t^2]=1, E[z_t^4]=m4
    results:
      rho_s(k) = rho_s(1) * kappa^(k-1),  k>=1
      kappa = alpha + beta
    requires:
      kappa < 1 (finite E[h])
      eta < 1 (finite E[h^2]) where eta = beta^2 + 2*alpha*beta + alpha^2*m4
      nu > 4 (finite m4 for t)
    """
    kappa = alpha + beta
    if kappa >= 1.0:
        raise ValueError("need alpha + beta < 1")
    m4 = m4_student_t_standardized(nu)
    eta = beta * beta + 2.0 * alpha * beta + (alpha * alpha) * m4
    if eta >= 1.0:
        raise ValueError("need eta < 1 for finite E[h^2] (finite fourth moment of eps)")
    hbar = omega / (1.0 - kappa)
    eh2 = (omega * omega) * (1.0 + kappa) / ((1.0 - kappa) * (1.0 - eta))
    var_s = m4 * eh2 - hbar * hbar
    cov1 = omega * hbar + (alpha * m4 + beta) * eh2 - hbar * hbar
    rho1 = cov1 / var_s
    rho = np.empty(nlags + 1, dtype=float)
    rho[0] = 1.0
    for k in range(1, nlags + 1):
        rho[k] = rho1 * (kappa ** (k - 1))
    info = {"kappa": kappa, "m4": m4, "eta": eta, "hbar": hbar, "eh2": eh2, "rho1": rho1}
    return rho, info
