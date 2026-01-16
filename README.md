# GARCH-from-scratch

Pure-Python (numpy/pandas) simulation and basic theory checks for:

- symmetric GARCH(1,1) with Student t innovations
- GJR-GARCH(1,1) (a.k.a. GJR / threshold GARCH) with Student t innovations

The scripts compare empirical autocorrelations from simulation to closed-form
theoretical autocorrelations of squared returns. The GJR script also compares
the leverage cross-correlation corr(eps_t, eps_{t+k}^2) to its theoretical
geometric decay (the level uses a simulation estimate of E[h^(3/2)]).

All formulas in the code assume symmetric innovations with E[z^2]=1.

## Files

- `garch.py`
  - core simulation functions
    - `simulate_garch_t(...)`
    - `simulate_gjr_garch_t(...)`
  - helper stats
    - `acf(...)`
    - `crosscorr(...)`
    - `m4_student_t_standardized(...)`
    - `abs_moment_student_t_standardized(...)`
  - theory
    - `theoretical_acf_sq_returns_garch(...)`
    - `theoretical_acf_sq_returns_gjr(...)`
    - `theoretical_crosscorr_return_future_sq(...)` (GJR leverage cross-corr)

- `xgarch.py`
  - main program for symmetric GARCH(1,1)
  - prints a table comparing theoretical vs empirical ACF of squared returns

- `xgarch_gjr.py`
  - main program for GJR-GARCH(1,1)
  - prints a table comparing theoretical vs empirical ACF of squared returns
  - prints a table comparing theoretical vs empirical corr(eps_t, eps_{t+k}^2)

## Quick start

Install dependencies:

```bash
pip install numpy pandas
```

Run symmetric GARCH(1,1):

```bash
python xgarch.py
```

Run GJR-GARCH(1,1):

```bash
python xgarch_gjr.py
```

Both scripts run multiple simulations with different seeds and print:

- the ACF comparison table(s) as pandas DataFrames
- an RMSE across lags as a quick "match quality" metric

## Model definitions

### Standardized Student t innovations

The code uses numpy's `standard_t(df=nu)` and rescales it so Var(z)=1:

- if x ~ t_nu, then Var(x) = nu/(nu-2) for nu>2
- set z = x * sqrt((nu-2)/nu) so Var(z)=1

This is what the simulators use in:

- `simulate_garch_t`
- `simulate_gjr_garch_t`

### Symmetric GARCH(1,1)

eps_t = sqrt(h_t) * z_t

h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

Stationarity (finite unconditional variance) requires:

kappa = alpha + beta < 1

### GJR-GARCH(1,1)

eps_t = sqrt(h_t) * z_t

h_t = omega + alpha * eps_{t-1}^2 + gamma * eps_{t-1}^2 * 1{eps_{t-1}<0} + beta * h_{t-1}

For symmetric z_t, P(eps_{t-1}<0)=1/2, so stationarity (finite unconditional
variance) requires:

kappa = alpha + beta + 0.5*gamma < 1

## Theoretical ACF of squared returns

Let s_t = eps_t^2. Under symmetric innovations with E[z^2]=1 and finite fourth
moment, the autocorrelation of s_t is geometric:

rho_s(k) = Corr(s_{t+k}, s_t) = rho_s(1) * kappa^(k-1),  k>=1

The code computes rho_s(1) in closed form by solving for E[h] and E[h^2].

### Fourth-moment condition

To have finite Var(eps_t^2) (and therefore a well-defined ACF of squared
returns), you also need a finite fourth moment and a contraction condition
for E[h^2]. The code checks this via `eta < 1`.

For standardized Student t, you need nu > 4 and:

m4 = E[z^4] = 3*(nu-2)/(nu-4)

#### Symmetric GARCH(1,1): eta

eta = beta^2 + 2*alpha*beta + alpha^2*m4

#### GJR-GARCH(1,1): eta

Define:

a2 = alpha^2 + alpha*gamma + 0.5*gamma^2

eta = beta^2 + 2*beta*(alpha + 0.5*gamma) + a2*m4

## Leverage cross-correlation for GJR

The GJR script also reports:

corr(eps_t, eps_{t+k}^2), k>=1

For symmetric z_t, this cross-correlation decays geometrically at the same
rate kappa. The level depends on E[h^(3/2)], which is not computed in closed
form in this repo; `xgarch_gjr.py` estimates it from the simulated path and
plugs it into `theoretical_crosscorr_return_future_sq`.

For symmetric GARCH(1,1), the corresponding leverage cross-correlations are
zero by symmetry, so `xgarch.py` does not compute them.

## Customizing runs

Edit the constants near the top of `main()` in `xgarch.py` or `xgarch_gjr.py`:

- `omega, alpha, beta, nu` (and `gamma` for GJR)
- `nobs` number of kept observations
- `burn` burn-in length
- `nlags` number of ACF lags to compare
- `nsim` number of simulations
- seed schedule (currently `seed0 + 1000000*i`)

## Notes

- This repo is intended for learning, sanity checks, and small experiments.
- Numerical safeguards:
  - `floor` prevents h_t from going non-positive due to rounding.
- If you choose parameters with kappa >= 1, the process has no finite
  unconditional variance; the theory checks in `theoretical_*` will raise.

## License

Add the license you prefer (MIT/BSD/Apache-2.0/etc.).
