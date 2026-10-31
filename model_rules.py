"""Manuscript protocol: replacement sampling, current support, end-MCS EMA.

Strategies use C=1 and D=0. All kernels mutate the supplied arrays in place.
The caller retains each experiment's payoff matrix and initialization scheme.
"""
import numpy as np
from numba import njit

PROTOCOL_ID = 'replacement_current_global_end_mcs_ema_v1'


@njit(cache=True)
def seed_model(seed):
    np.random.seed(seed)


@njit(cache=True)
def periodic_neighbors(L):
    result = np.empty((L * L, 4), dtype=np.int64)
    for x in range(L * L):
        row, col = x // L, x % L
        result[x, 0] = ((row - 1) % L) * L + col
        result[x, 1] = ((row + 1) % L) * L + col
        result[x, 2] = row * L + (col - 1) % L
        result[x, 3] = row * L + (col + 1) % L
    return result


@njit(cache=True)
def current_weights(strategies, reputations, epsilon):
    total, cooperative = 0.0, 0.0
    for x in range(len(strategies)):
        weight = reputations[x] + epsilon
        total += weight
        if strategies[x] == 1:
            cooperative += weight
    return total, cooperative


@njit(cache=True)
def fermi_probability(focal_fitness, neighbor_fitness, noise):
    # Stable logistic: no hard truncation of finite probabilities.
    exponent = (focal_fitness - neighbor_fitness) / noise
    if exponent >= 0.0:
        value = np.exp(-exponent)
        return value / (1.0 + value)
    return 1.0 / (1.0 + np.exp(exponent))


@njit(cache=True)
def fixed_fitness(x, strategies, reputations, neighbors, total, cooperative,
                  alpha, beta, epsilon, reward, sucker, temptation, punishment):
    strategy = strategies[x]
    numerator = reputations[x] + epsilon
    denominator = numerator
    payoff = 0.0
    for edge in range(neighbors.shape[1]):
        y = neighbors[x, edge]
        weight = reputations[y] + epsilon
        denominator += weight
        if strategies[y] == strategy:
            numerator += weight
        if strategy == 1:
            payoff += reward if strategies[y] == 1 else sucker
        else:
            payoff += temptation if strategies[y] == 1 else punishment
    if beta == 0.0:
        return payoff
    # Only remove roundoff outside [0,1], not genuine small support.
    global_c = min(1.0, max(0.0, cooperative / total))
    support = alpha * numerator / denominator + (1.0 - alpha) * (
        global_c if strategy == 1 else 1.0 - global_c)
    return payoff * support ** beta


@njit(cache=True)
def fixed_event(x, strategies, reputations, neighbors, total, cooperative,
                noise, alpha, beta, epsilon, u, reward, sucker, temptation, punishment):
    old = strategies[x]
    new = old
    if np.random.random() < u:
        new = np.random.randint(0, 2)
    else:
        y = neighbors[x, np.random.randint(0, neighbors.shape[1])]
        if old != strategies[y]:
            focal = fixed_fitness(x, strategies, reputations, neighbors, total,
                                  cooperative, alpha, beta, epsilon,
                                  reward, sucker, temptation, punishment)
            target = fixed_fitness(y, strategies, reputations, neighbors, total,
                                   cooperative, alpha, beta, epsilon,
                                   reward, sucker, temptation, punishment)
            if np.random.random() < fermi_probability(focal, target, noise):
                new = strategies[y]
    if new != old:
        strategies[x] = new
        cooperative += (reputations[x] + epsilon) * (new - old)
    return cooperative


@njit(cache=True)
def mcs_fixed(strategies, reputations, neighbors, noise, b, alpha, beta,
              gamma, epsilon, u, reward=1.0, sucker=0.0, punishment=0.0):
    total, cooperative = current_weights(strategies, reputations, epsilon)
    for _ in range(len(strategies)):
        x = np.random.randint(0, len(strategies))
        cooperative = fixed_event(x, strategies, reputations, neighbors,
                                  total, cooperative, noise, alpha, beta,
                                  epsilon, u, reward, sucker, b, punishment)
    for x in range(len(strategies)):
        reputations[x] = (1.0 - gamma) * reputations[x] + gamma * strategies[x]


@njit(cache=True)
def csr_fitness(x, strategies, reputations, offsets, neighbors, total, cooperative,
                alpha, beta, epsilon, reward, sucker, temptation, punishment):
    strategy = strategies[x]
    numerator = reputations[x] + epsilon
    denominator = numerator
    payoff = 0.0
    for edge in range(offsets[x], offsets[x + 1]):
        y = neighbors[edge]
        weight = reputations[y] + epsilon
        denominator += weight
        if strategies[y] == strategy:
            numerator += weight
        if strategy == 1:
            payoff += reward if strategies[y] == 1 else sucker
        else:
            payoff += temptation if strategies[y] == 1 else punishment
    if beta == 0.0:
        return payoff
    global_c = min(1.0, max(0.0, cooperative / total))
    support = alpha * numerator / denominator + (1.0 - alpha) * (
        global_c if strategy == 1 else 1.0 - global_c)
    return payoff * support ** beta


@njit(cache=True)
def csr_event(x, strategies, reputations, offsets, neighbors, total, cooperative,
              noise, alpha, beta, epsilon, u, reward, sucker, temptation, punishment):
    old = strategies[x]
    new = old
    if np.random.random() < u:
        new = np.random.randint(0, 2)
    else:
        degree = offsets[x + 1] - offsets[x]
        if degree == 0:
            return cooperative
        y = neighbors[offsets[x] + np.random.randint(0, degree)]
        if old != strategies[y]:
            focal = csr_fitness(x, strategies, reputations, offsets, neighbors,
                                total, cooperative, alpha, beta, epsilon,
                                reward, sucker, temptation, punishment)
            target = csr_fitness(y, strategies, reputations, offsets, neighbors,
                                 total, cooperative, alpha, beta, epsilon,
                                 reward, sucker, temptation, punishment)
            if np.random.random() < fermi_probability(focal, target, noise):
                new = strategies[y]
    if new != old:
        strategies[x] = new
        cooperative += (reputations[x] + epsilon) * (new - old)
    return cooperative


@njit(cache=True)
def mcs_csr(strategies, reputations, offsets, neighbors, noise, b, alpha, beta,
            gamma, epsilon, u, reward=1.0, sucker=0.0, punishment=0.0):
    total, cooperative = current_weights(strategies, reputations, epsilon)
    for _ in range(len(strategies)):
        x = np.random.randint(0, len(strategies))
        cooperative = csr_event(x, strategies, reputations, offsets, neighbors,
                                total, cooperative, noise, alpha, beta, epsilon,
                                u, reward, sucker, b, punishment)
    for x in range(len(strategies)):
        reputations[x] = (1.0 - gamma) * reputations[x] + gamma * strategies[x]


_SQUARE_MAPS = {}


def mcs_square(strategies, reputations, L, noise, b, alpha, beta, gamma, epsilon, u):
    if L not in _SQUARE_MAPS:
        _SQUARE_MAPS[L] = periodic_neighbors(L)
    mcs_fixed(strategies.reshape(L * L), reputations.reshape(L * L),
              _SQUARE_MAPS[L], noise, b, alpha, beta, gamma, epsilon, u)


def prepare_run_directory(script_path, parameters=None):
    """Keep new relative outputs separate from all historical result files."""
    from pathlib import Path
    from datetime import datetime
    import os, json, hashlib, platform, sys
    import numba
    script = Path(script_path).resolve()
    root = script.parent / ('results_' + PROTOCOL_ID)
    output = root / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol=PROTOCOL_ID, source=str(script), arguments=sys.argv[1:],
                    source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                    core_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    python=platform.python_version(), numpy=np.__version__, numba=numba.__version__)
    if parameters is not None:
        captured = {}
        for name, value in parameters.items():
            if name.isupper() and isinstance(value, (str, int, float, bool, list, tuple, Path, np.ndarray)):
                captured[name] = value.tolist() if isinstance(value, np.ndarray) else str(value) if isinstance(value, Path) else value
        manifest['script_defaults'] = captured
    (output / 'protocol_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.chdir(output)
    return output
