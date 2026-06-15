"""
Fixture generator for PolyTrend tests.

Run this script to (re)generate all CSV fixtures used by the test suite.
Each fixture is generated from a known ground-truth polynomial so that
test assertions can be written against closed-form expected values.

Convention: filenames encode the ground truth so tests remain self-documenting.
  linear_clean      → y = 2x + 1,         no noise, no errors
  quadratic_clean   → y = x² - 3x + 2,    no noise, no errors
  cubic_clean       → y = x³ - x² + x - 1, no noise, no errors
  linear_noisy      → y = 2x + 1 + N(0,5)
  quadratic_noisy   → y = x² - 3x + 2 + N(0,10)
  uniform_errors    → y = 2x + 1, homogeneous error column (σ=1)
  mixed_errors      → y = x² - 3x + 2, heterogeneous error column
  two_points        → minimum valid input (n=2)
  constant          → y = 7 for all x (degree-0 signal; tests BIC on degenerate case)
  large             → n=200, quadratic, for performance pinning
  negative_x        → x in [-50, 50], linear
  duplicate_x       → repeated x values (sklearn handles this; we pin the behaviour)
"""

import csv
import random
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
FIXTURES_DIR.mkdir(exist_ok=True)

random.seed(42)  # reproducible noise


def write_csv(name: str, rows: list[tuple], header: tuple = ('x', 'f(x)')) -> Path:
  path = FIXTURES_DIR / f'{name}.csv'
  with path.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
  print(f'  wrote {path.name}  ({len(rows)} rows)')
  return path


def noisy(scale: float) -> float:
  return random.gauss(0, scale)


# ── clean polynomial fixtures (oracle values are exact) ─────────────────────

write_csv(
  'linear_clean',
  [(x, 2 * x + 1) for x in range(1, 21)],
)

write_csv(
  'quadratic_clean',
  [(x, x**2 - 3 * x + 2) for x in range(1, 21)],
)

write_csv(
  'cubic_clean',
  [(x, x**3 - x**2 + x - 1) for x in range(1, 21)],
)

# ── noisy fixtures (oracle degree is still known; exact values are not) ──────

write_csv(
  'linear_noisy',
  [(x, 2 * x + 1 + noisy(5)) for x in range(1, 51)],
)

write_csv(
  'quadratic_noisy',
  [(x, x**2 - 3 * x + 2 + noisy(10)) for x in range(1, 51)],
)

# ── error-column fixtures ────────────────────────────────────────────────────

write_csv(
  'uniform_errors',
  [(x, 2 * x + 1, 1.0) for x in range(1, 21)],
  header=('x', 'f(x)', 'err'),
)

write_csv(
  'mixed_errors',
  [(x, x**2 - 3 * x + 2, abs(noisy(2)) + 0.1) for x in range(1, 21)],
  header=('x', 'f(x)', 'err'),
)

# ── edge / atypical fixtures ─────────────────────────────────────────────────

write_csv(
  'two_points',
  [(1, 3), (2, 5)],  # exactly y = 2x + 1
)

write_csv(
  'constant',
  [(x, 7.0) for x in range(1, 11)],
)

write_csv(
  'large',
  [(x, x**2 - 3 * x + 2 + noisy(5)) for x in range(1, 201)],
)

write_csv(
  'negative_x',
  [(x, 2 * x + 1) for x in range(-50, 51) if x != 0],
)

write_csv(
  'duplicate_x',
  [(x % 5 + 1, 2 * (x % 5 + 1) + 1 + noisy(0.5)) for x in range(20)],
)

print('Done.')
