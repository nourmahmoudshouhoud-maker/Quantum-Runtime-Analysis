import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define runtime functions for Shor and Grover
# (Classical and Quantum)
def classical_runtime_shor_logx(x, P=0):
    n = 10**x
    if n <= 1: return np.inf
    term1 = (64/9 * n)**(1/3)
    term2 = np.log(n)**(2/3)
    return np.log10(np.exp(term1 * term2) / (10**P))

def quantum_runtime_shor_logx(x):
    n = 10**x
    if n <= 1: return np.inf
    return np.log10((n**2) * np.log(n))

def classical_runtime_grover_logx(x, P=0):
    n = 10**x
    if n <= 0: return np.inf
    return np.log10(n / (10**P))

def quantum_runtime_grover_logx(x):
    n = 10**x
    if n <= 0: return np.inf
    return np.log10(np.sqrt(n))

# Parameter setup for quantum roadmap and hardware
def define_parameters(problem_type="shor", roadmap_data=None,
    hws_initial=3.78, qir_annual_percent=-10, plqr_initial=264,
    rir_annual_percent=-23, cir_annual_percent=10,
    connectivity_penalty_func=lambda q: np.sqrt(q),
    num_classical_processors_log10=8):

    if roadmap_data is None:
        roadmap_data = {2020: 27, 2024: 133, 2025: 156, 2028: 1092}

    params = {
        "problem_type": problem_type,
        "roadmap_data": roadmap_data,
        "hws_initial": hws_initial,
        "qir_annual_percent": qir_annual_percent,
        "plqr_initial": plqr_initial,
        "rir_annual_percent": rir_annual_percent,
        "cir_annual_percent": cir_annual_percent,
        "connectivity_penalty_func": connectivity_penalty_func,
        "num_classical_processors_log10": num_classical_processors_log10,
    }
    if problem_type == "shor":
        params["classical_runtime_logx"] = classical_runtime_shor_logx
        params["quantum_runtime_logx"] = quantum_runtime_shor_logx
    elif problem_type == "grover":
        params["classical_runtime_logx"] = classical_runtime_grover_logx
        params["quantum_runtime_logx"] = quantum_runtime_grover_logx
    return params

BASE_YEAR = 2020

# Roadmap and hardware functions
def get_roadmap_qubits(year, params):
    years = sorted(params["roadmap_data"].keys())
    qubits = [params["roadmap_data"][y] for y in years]
    if year in years: return params["roadmap_data"][year]
    if year < years[0]:
        if len(years) < 2: return qubits[0]
        rate = (np.log(qubits[1]) - np.log(qubits[0])) / (years[1] - years[0])
        return float(np.exp(np.log(qubits[0]) + rate * (year - years[0])))
    else:
        if len(years) < 2: return qubits[-1]
        rate = (np.log(qubits[-1]) - np.log(qubits[-2])) / (years[-1] - years[-2])
        return float(np.exp(np.log(qubits[-1]) + rate * (year - years[-1])))

def calculate_hws(year, params, base_year=BASE_YEAR):
    qir_factor = (1 + params["qir_annual_percent"]/100)**(year - base_year)
    return params["hws_initial"] * qir_factor

def calculate_plqr(year, params, base_year=BASE_YEAR):
    rir_factor = (1 + params["rir_annual_percent"]/100)**(year - base_year)
    return max(3, params["plqr_initial"] * rir_factor)

def adv_logx(x, year, params):
    x = np.asarray(x, dtype=float).item()
    hws_t = calculate_hws(year, params)
    cir_factor = (1 + params["cir_annual_percent"]/100)**(year - BASE_YEAR)
    num_processors = 10**params["num_classical_processors_log10"] / cir_factor
    classical_rt = params["classical_runtime_logx"](x, np.log10(num_processors))
    quantum_rt = params["quantum_runtime_logx"](x) + np.log10(params["connectivity_penalty_func"](10**x))
    return classical_rt - (quantum_rt + hws_t)

def find_n_star(year, params, x_guess=2):
    func = lambda xx: adv_logx(xx, year, params)
    try:
        root = fsolve(func, x_guess, maxfev=2000)
        return float(root[0]) if root[0] > 0 else np.nan
    except: return np.nan

# Plotting functions
def plot_n_star_and_runtimes(params, x_range=(0, 6), base_year=2025):
    xs = np.linspace(x_range[0], x_range[1], 300)
    classical = [params["classical_runtime_logx"](x) for x in xs]
    quantum = [params["quantum_runtime_logx"](x) + calculate_hws(base_year, params) for x in xs]
    plt.figure(figsize=(8,5))
    plt.plot(xs, classical, label="Classical (log10 runtime)")
    plt.plot(xs, quantum, label="Quantum (log10 runtime + HWS)")
    try:
        x_star = find_n_star(base_year, params, np.mean(x_range))
        if not np.isnan(x_star):
            plt.axvline(x=x_star, color="red", linestyle=":", label=f"log10(n*)={x_star:.2f}")
    except: pass
    plt.xlabel("log10(Problem size n)")
    plt.ylabel("log10(Runtime)")
    plt.legend(); plt.grid(True); plt.show()

def plot_qea_over_time(params, years_range=(2020, 2040), n_target=2048):
    x_target = np.log10(np.asarray(n_target, dtype=float))
    years = np.arange(years_range[0], years_range[1]+1)
    feasibility, advantage = [], []
    for y in years:
        R = get_roadmap_qubits(y, params)
        plqr = calculate_plqr(y, params)
        logical_qubits = max(R / plqr, 1e-12)
        feasibility.append(np.log10(logical_qubits))
        advantage.append(find_n_star(y, params, x_target))
    plt.figure(figsize=(8,5))
    plt.plot(years, feasibility, label="Feasibility (max log10(n))")
    plt.plot(years, advantage, label="Advantage (min log10(n*))")
    plt.axhline(y=x_target, color="purple", linestyle="--", label=f"Target log10(n)={x_target:.2f}")
    plt.xlabel("Year"); plt.ylabel("log10(Problem size n)")
    plt.legend(); plt.grid(True); plt.show()

# Example usage for Shor and Grover
shor_params = define_parameters("shor")
plot_n_star_and_runtimes(shor_params, x_range=(0, 4), base_year=2025)
plot_qea_over_time(shor_params, years_range=(2020, 2040), n_target=2048)

grover_params = define_parameters("grover", roadmap_data={2020:32, 2024:256, 2028:1024},
    hws_initial=8.48, plqr_initial=32, connectivity_penalty_func=lambda q: 1)
plot_n_star_and_runtimes(grover_params, x_range=(0, 25), base_year=2025)
plot_qea_over_time(grover_params, years_range=(2020, 2050), n_target=10**20)
