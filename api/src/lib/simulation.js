// src/lib/simulation.js

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function computeBudgetQuality(capex, subsidy, deficitTarget, credibility) {
  const capexScore = Math.min(capex / 12.0, 1.0);
  const subsidyScore = Math.min(subsidy / 5.0, 1.0);
  const deficitScore = Math.max(0, 1 - (deficitTarget - 3.0) / 7.0);
  const raw = 0.5 * capexScore + 0.25 * subsidyScore + 0.25 * deficitScore;
  let quality = raw * (0.6 + 0.4 * credibility);
  quality = Math.min(Math.max(quality, 0.05), 1.0);
  return quality;
}

export function simulate(budgetQuality, credibility, initialBelief, rounds = 30, beta = 0.2, frontloadBoost = 0, noise = 0.15) {
  let p = Math.min(1.0, Math.max(0.0, initialBelief + frontloadBoost));
  const investRates = [];
  const beliefHistory = [p];
  const base = budgetQuality * 10;

  // 🔥 CREDIBILITY TRAP – payoffs flip below 0.5
  let payoffGood, payoffBad, payoffMismatch;
  if (credibility >= 0.5) {
    payoffGood = 0.58 * base;
    payoffBad = 0.42 * base;
    payoffMismatch = 0.12 * base;
  } else {
    payoffGood = 0.08 * base;   // very low – good equilibrium unattractive
    payoffBad = 0.52 * base;
    payoffMismatch = 0.04 * base;
  }

  for (let t = 0; t < rounds; t++) {
    const eInvest = p * payoffGood + (1 - p) * payoffMismatch;
    const eHold   = p * payoffMismatch + (1 - p) * payoffBad;
    let diff = eInvest - eHold;
    let prob = sigmoid(diff);
    prob = prob + (Math.random() - 0.5) * noise;
    prob = Math.min(0.95, Math.max(0.05, prob));
    const invest = Math.random() < prob ? 1 : 0;
    investRates.push(invest);
    p = (1 - beta) * p + beta * invest;
    p = Math.min(1.0, Math.max(0.0, p));
    beliefHistory.push(p);
  }

  const finalRate = investRates.slice(-10).reduce((a, b) => a + b, 0) / 10 * 100;
  const finalBelief = beliefHistory[beliefHistory.length - 1];
  let convergeRound = rounds;
  for (let i = rounds - 10; i < rounds - 1; i++) {
    if (Math.abs(beliefHistory[i + 1] - beliefHistory[i]) < 0.005) {
      convergeRound = i + 1;
      break;
    }
  }
  const equilibrium = finalRate > 55 ? "Good (Invest, Invest)" : finalRate < 45 ? "Bad (Hold, Hold)" : "Multiple NE";

  return { finalRate, finalBelief, convergeRound, equilibrium, payoffGood, payoffBad, payoffMismatch };
}