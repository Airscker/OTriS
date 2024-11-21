import jax
import netket as nk
import os
import matplotlib.pyplot as plt
import json

# Ensure all logs are saved in a separate directory
log_dir = "gp_logs"
os.makedirs(log_dir, exist_ok=True)

# 1D Lattice
L = 16
Vp = 2.0

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Loop over gp values
for gp in [0.02 * i for i in range(151)]:
    # The Hamiltonian
    ha = nk.operator.LocalOperator(hi)
    # List of dissipative jump operators
    j_ops = []
    # Observables
    obs_sx = nk.operator.LocalOperator(hi)
    obs_sy = nk.operator.LocalOperator(hi, dtype=complex)
    obs_sz = nk.operator.LocalOperator(hi)

    for i in range(L):
        ha += (gp / 2.0) * nk.operator.spin.sigmax(hi, i)
        ha += (
            (Vp / 4.0)
            * nk.operator.spin.sigmaz(hi, i)
            * nk.operator.spin.sigmaz(hi, (i + 1) % L)
        )
        # sigma_{-} dissipation on every site
        j_ops.append(nk.operator.spin.sigmam(hi, i))
        obs_sx += nk.operator.spin.sigmax(hi, i)
        obs_sy += nk.operator.spin.sigmay(hi, i)
        obs_sz += nk.operator.spin.sigmaz(hi, i)

    # Create the Liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)

    # Model and Sampler
    ma = nk.models.NDM(beta=1)
    sa = nk.sampler.MetropolisLocal(lind.hilbert)

    # Optimizer
    op = nk.optimizer.Sgd(0.01)
    sr = nk.optimizer.SR(diag_shift=0.01)

    # Variational State
    vs = nk.vqs.MCMixedState(sa, ma, n_samples=2000, n_samples_diag=512)
    vs.init_parameters(jax.nn.initializers.normal(stddev=0.01))

    # Steady state solver
    ss = nk.SteadyState(lind, op, variational_state=vs, preconditioner=sr)

    # Observables
    obs = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}

    # Run the optimization and save log to a unique file
    out_file = os.path.join(log_dir, f"test_gp_{gp:.2f}")
    ss.run(n_iter=1500, out=out_file, obs=obs)

# Plotting the results
gp_values = [0.02 * i for i in range(151)]
sx_means = []
sy_means = []
sz_means = []

# Extract data from log files
for gp in gp_values:
    log_file = os.path.join(log_dir, f"test_gp_{gp:.2f}.log")
    with open(log_file, "r") as f:
        data = json.load(f)
        sx_means.append(data["Sx"]["Mean"]["real"][-1] / L)
        sy_means.append(data["Sy"]["Mean"]["real"][-1] / L)
        sz_means.append(data["Sz"]["Mean"]["real"][-1] / L)

# Plot Sx
plt.figure()
plt.plot(gp_values, sx_means, label="Sx Mean", marker='o')
plt.xlabel("gp")
plt.ylabel("Sx Mean (Last Iteration)")
plt.title("Sx Mean vs gp")
plt.legend()
plt.grid()
plt.savefig("Sx_vs_gp.png")
plt.show()

# Plot Sy
plt.figure()
plt.plot(gp_values, sy_means, label="Sy Mean", marker='o')
plt.xlabel("gp")
plt.ylabel("Sy Mean (Last Iteration)")
plt.title("Sy Mean vs gp")
plt.legend()
plt.grid()
plt.savefig("Sy_vs_gp.png")
plt.show()

# Plot Sz
plt.figure()
plt.plot(gp_values, sz_means, label="Sz Mean", marker='o')
plt.xlabel("gp")
plt.ylabel("Sz Mean (Last Iteration)")
plt.title("Sz Mean vs gp")
plt.legend()
plt.grid()
plt.savefig("Sz_vs_gp.png")
plt.show()
