from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "BudgetGame API running"}


# ── Request model ──────────────────────────────────────────────────────────────
class SimParams(BaseModel):
    capex:          float = 11.1
    subsidy:        float = 3.8
    deficit_target: float = 5.1
    credibility:    float = 0.70  # [0, 1] — not raw delivery rate
    n_rounds:       int   = 80
    front_loading:  bool  = False  # if True, credibility is persistently boosted
    n_seeds:        int   = 50


# ── Budget quality ─────────────────────────────────────────────────────────────
def compute_budget_quality(capex: float, subsidy: float,
                           deficit_target: float, credibility: float) -> float:
    """Composite score in [0.05, 1.0]. Credibility scales the raw score."""
    capex_score   = min(capex   / 12.0, 1.0)
    subsidy_score = min(subsidy /  5.0, 1.0)
    deficit_score = max(0.0, 1.0 - (deficit_target - 3.0) / 7.0)
    raw     = 0.5 * capex_score + 0.25 * subsidy_score + 0.25 * deficit_score
    quality = raw * (0.6 + 0.4 * credibility)
    return round(float(np.clip(quality, 0.05, 1.0)), 3)


# ── Front-loading: persistent credibility boost ────────────────────────────────
AVG_DELIVERY = 0.926  # India historical avg capex delivery rate 2018-2025

def apply_front_loading(credibility: float) -> float:
    """
    Front-loading H1 capex is a costly, observable action that PERSISTENTLY
    raises effective credibility for the rest of the fiscal year.

    Theoretical grounding (Spence signalling): signals are credible only when
    costly and hard to reverse. Deploying 60%+ of capex in H1 commits funds
    irreversibly, so it shifts the effective credibility parameter that governs
    BOTH payoffs AND belief updating — not just a one-time belief nudge.

    Boost = (avg delivery rate - 50% baseline) * 0.30
          = (0.926 - 0.50) * 0.30
          = ~0.128

    Attenuated by 0.30 so the boost is meaningful (~13pp) but cannot rescue
    a structurally bad budget (credibility + 0.128 still leaves low-cred
    scenarios below the high-cred equilibrium).
    """
    boost = (AVG_DELIVERY - 0.50) * 0.30  # ~0.128
    return float(np.clip(credibility + boost, 0.0, 1.0))


# ── Payoffs (continuous, no hard threshold) ────────────────────────────────────
def get_payoffs(budget_quality: float, credibility: float):
    """
    Unit-scale payoffs in [0, 1]. Large separation ensures trap is visible.
    budget_quality scales magnitude; credibility determines direction.
    """
    scale           = 0.5 + 0.5 * budget_quality
    payoff_good     = scale * (0.10 + 0.80 * credibility)  # 0.05 -> 0.85
    payoff_bad      = scale * (0.75 - 0.65 * credibility)  # 0.75 -> 0.10
    payoff_mismatch = scale * 0.15
    return payoff_good, payoff_bad, payoff_mismatch


# ── Belief update (separate noisy signal, not same payoffs) ───────────────────
def update_belief(p: float, invest: int, credibility: float,
                  noise: float = 0.25) -> float:
    """
    Noisy binary signal Bayesian update.
    Uses a SEPARATE signal structure from decision payoffs so that p
    does not collapse to the same payoff-attractor regardless of start.
    """
    lh_cred     = (1 - noise) if invest == 1 else noise
    lh_not_cred = noise       if invest == 1 else (1 - noise)
    lh_cred     *= credibility
    lh_not_cred *= (1 - credibility)
    numerator    = lh_cred * p
    denominator  = lh_cred * p + lh_not_cred * (1 - p)
    if denominator < 1e-9:
        return p
    return float(np.clip(numerator / denominator, 0.0, 1.0))


# ── Core simulation ────────────────────────────────────────────────────────────
def simulate_player(budget_quality: float, credibility: float,
                    initial_belief: float, n_rounds: int = 80,
                    n_seeds: int = 50, noise: float = 0.25):
    """
    Returns (mean_rate%, sd_rate%, smoothed_series).
    Credibility passed in is ALREADY adjusted for front-loading upstream.
    Equilibrium measured over last 25 rounds after burn-in.
    """
    payoff_good, payoff_bad, payoff_mismatch = get_payoffs(budget_quality, credibility)
    all_series  = []
    final_rates = []

    for seed in range(n_seeds):
        rng    = np.random.default_rng(seed)
        p      = initial_belief
        series = []

        for _ in range(n_rounds):
            e_invest = p * payoff_good     + (1 - p) * payoff_mismatch
            e_hold   = p * payoff_mismatch + (1 - p) * payoff_bad
            diff     = e_invest - e_hold
            prob     = 1.0 / (1.0 + np.exp(-4.0 * diff))
            prob     = float(np.clip(prob, 0.05, 0.95))
            invest   = int(rng.random() < prob)
            series.append(invest)
            p = update_belief(p, invest, credibility, noise)

        all_series.append(series)
        final_rates.append(float(np.mean(series[-25:])) * 100)

    mean_rate      = float(np.mean(final_rates))
    sd_rate        = float(np.std(final_rates))
    mean_traj      = np.mean(all_series, axis=0).tolist()
    chunk          = max(1, n_rounds // 10)
    smoothed       = [
        round(float(np.mean(mean_traj[i: i + chunk])) * 100, 1)
        for i in range(0, n_rounds, chunk)
    ]
    return round(mean_rate, 1), round(sd_rate, 1), smoothed


# ── Initial belief per player ──────────────────────────────────────────────────
def get_initial_belief(credibility: float, quality_signal: float,
                       base_optimism: float = 0.5) -> float:
    """Credibility passed in is already front-loading adjusted if applicable."""
    belief = base_optimism + 0.4 * (credibility - 0.7) + 0.3 * (quality_signal - 0.5)
    return float(np.clip(belief, 0.05, 0.95))


# ── Payoff matrix & Nash ───────────────────────────────────────────────────────
def compute_payoff_matrix(bq: float, credibility: float) -> dict:
    b = bq * 10.0
    return {
        'high_capex_invest': (round(b * credibility, 1), round(b * 0.90, 1)),
        'high_capex_hold':   (round(b * 0.35,        1), round(b * 0.55, 1)),
        'austerity_invest':  (round(b * 0.45,        1), round(b * 0.25, 1)),
        'austerity_hold':    (round(b * 0.50,        1), round(b * 0.45, 1)),
    }


def find_nash(p: dict) -> list:
    hci, hch = p['high_capex_invest'], p['high_capex_hold']
    ai,  ah  = p['austerity_invest'],  p['austerity_hold']
    nash = []
    if hci[0] >= ai[0]  and hci[1] >= hch[1]: nash.append('high_capex_invest')
    if hch[0] >= ah[0]  and hch[1] >= hci[1]: nash.append('high_capex_hold')
    if ai[0]  >= hci[0] and ai[1]  >= ah[1]:  nash.append('austerity_invest')
    if ah[0]  >= hch[0] and ah[1]  >= ai[1]:  nash.append('austerity_hold')
    return nash


# ── /simulate ─────────────────────────────────────────────────────────────────
@app.post("/simulate")
def simulate(p: SimParams):
    # KEY FIX: if front_loading, boost credibility ONCE here, upstream of
    # everything. This means payoffs, belief updating, budget quality, Nash
    # payoff matrix — all respond to the higher credibility, not just beliefs.
    # This is what makes front-loading a persistent signal, not a cheap nudge.
    effective_cred = apply_front_loading(p.credibility) if p.front_loading \
                     else p.credibility

    bq = compute_budget_quality(p.capex, p.subsidy, p.deficit_target, effective_cred)

    # Player initial beliefs (use effective_cred so beliefs are also consistent)
    corp_init  = get_initial_belief(effective_cred, bq,            base_optimism=0.50)
    rural_init = get_initial_belief(effective_cred, bq * 0.85,     base_optimism=0.45)
    bond_init  = get_initial_belief(
        0.5,
        float(np.clip((p.deficit_target - 3) / 7 + 0.2, 0, 1)),
        base_optimism=0.55
    )
    state_init = get_initial_belief(effective_cred * 0.85, bq * 0.80, base_optimism=0.40)

    # Simulations
    corp_rate,  corp_sd,  corp_series  = simulate_player(
        bq,        effective_cred,        corp_init,
        n_rounds=p.n_rounds, n_seeds=p.n_seeds)
    rural_rate, rural_sd, rural_series = simulate_player(
        bq * 0.85, effective_cred,        rural_init,
        n_rounds=p.n_rounds, n_seeds=p.n_seeds)
    bond_rate,  bond_sd,  bond_series  = simulate_player(
        float(np.clip((p.deficit_target - 3) / 7 + 0.2, 0.05, 1.0)),
        0.5, bond_init,
        n_rounds=p.n_rounds, n_seeds=p.n_seeds)
    state_rate, state_sd, state_series = simulate_player(
        bq * 0.80, effective_cred * 0.85, state_init,
        n_rounds=p.n_rounds, n_seeds=p.n_seeds)

    # Compute lift vs baseline (no front-loading) for corporates
    if p.front_loading:
        bq_base       = compute_budget_quality(p.capex, p.subsidy,
                                               p.deficit_target, p.credibility)
        corp_init_base = get_initial_belief(p.credibility, bq_base, 0.50)
        corp_base, _, _ = simulate_player(bq_base, p.credibility, corp_init_base,
                                          n_rounds=p.n_rounds, n_seeds=p.n_seeds)
        fl_lift = round(corp_rate - corp_base, 1)
    else:
        fl_lift = None

    pay  = compute_payoff_matrix(bq, effective_cred)
    nash = find_nash(pay)
    gdp  = round(
        1.0
        + corp_rate  / 100 * 0.5
        + rural_rate / 100 * 0.3
        + (1 - bond_rate / 100) * 0.2
        + state_rate / 100 * 0.3,
        2
    )

    return {
        'budget_quality':        bq,
        'effective_credibility': round(effective_cred, 3),
        'gdp_multiplier':        gdp,
        'front_loading_lift_pp': fl_lift,
        'crowding_out': (
            'High' if p.deficit_target > 6.5
            else 'Med' if p.deficit_target > 4.5
            else 'Low'
        ),
        'equilibrium_type': (
            'Nash (Pure)'  if len(nash) == 1
            else 'Multiple NE' if len(nash) > 1
            else 'No Pure NE'
        ),
        'nash_cells': nash,
        'payoffs':    pay,
        'players': {
            'corporates':  {'final': corp_rate,  'sd': corp_sd,
                            'series': corp_series},
            'rural':       {'final': rural_rate, 'sd': rural_sd,
                            'series': rural_series},
            'bond_market': {'final': bond_rate,  'sd': bond_sd,
                            'series': bond_series},
            'states':      {'final': state_rate, 'sd': state_sd,
                            'series': state_series},
        },
    }


# ── /presets ──────────────────────────────────────────────────────────────────
@app.get("/presets")
def presets():
    return [
        {'name': 'India FY25',       'icon': '🇮🇳',
         'params': {'capex': 11.1, 'subsidy': 3.8,
                    'deficit_target': 5.1,  'credibility': 0.75}},
        {'name': 'COVID shock',      'icon': '🦠',
         'params': {'capex':  4.4, 'subsidy': 6.2,
                    'deficit_target': 9.2,  'credibility': 0.30}},
        {'name': 'Austerity',        'icon': '📉',
         'params': {'capex':  5.0, 'subsidy': 2.0,
                    'deficit_target': 3.0,  'credibility': 0.80}},
        {'name': 'Max stimulus',     'icon': '🚀',
         'params': {'capex': 14.0, 'subsidy': 7.0,
                    'deficit_target': 7.5,  'credibility': 0.55}},
        {'name': 'Credibility trap', 'icon': '🪤',
         'params': {'capex': 10.0, 'subsidy': 4.0,
                    'deficit_target': 5.5,  'credibility': 0.25}},
    ]