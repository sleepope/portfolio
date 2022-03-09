from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from stock import Stock

plt.style.use('fivethirtyeight')


def solve(A, r_F, mu_R, sigma_R, verbose=False):
    mu = r_F + (mu_R - r_F)**2 / (A * sigma_R**2)
    sigma = (mu_R - r_F) / (A * sigma_R)
    u = mu - A * sigma**2 / 2
    sr = (mu - r_F) / sigma

    if verbose:
        print("===== NUMERICAL RESULT =====")
        print(f"mu*\t= {mu:.4%}")
        print(f"sigma*\t= {sigma:.4%}")
        print(f"U*\t= {u:.4%}")
        print(f"sharpe\t= {sr:.4f}")
        print()

    return mu, sigma, u, sr


def plot(A, r_F, mu_R, sigma_R):
    mu, sigma, u, sr = solve(A, r_F, mu_R, sigma_R)

    def func(x):
        return A / 2 * x**2 + u

    def cons(x):
        return r_F + x * (mu_R - r_F) / sigma_R

    fig, ax = plt.subplots()
    ticks = np.linspace(-0.005, 0.015, 100)

    ax.plot(ticks, func(ticks), linewidth=3)
    ax.plot(ticks, cons(ticks), linewidth=3)
    ax.scatter(sigma, mu, marker='x', c='black')
    ax.legend(['Utility Function', 'CAL', 'Optimal Point'])

    ax.set_title('Optimal Allocation')
    ax.set_xlabel('Risk (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.set_xticklabels([f'{100*x:.2}' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{100*y:.2}' for y in ax.get_yticks()])

    fig.tight_layout()
    fig.savefig('./allocation.png', dpi=300)


def main():
    # load moutai data
    moutai = Stock("./data", '600519.XSHG', 2021)
    mu_R = moutai.mean
    sigma_R = moutai.std

    # other constants
    A = 28 / 9
    r_F_year = 0.025
    r_F = (r_F_year + 1)**(1 / 365) - 1

    # solve & plot
    solve(A, r_F, mu_R, sigma_R, verbose=True)
    plot(A, r_F, mu_R, sigma_R)


if __name__ == "__main__":
    main()
