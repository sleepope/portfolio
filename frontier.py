import os
import pickle
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
from stock import Stock

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s',
                    level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_home", type=str, default='./data')
    parser.add_argument("--data_year", type=int, default=2021)
    parser.add_argument("--data_filename", type=str, default='./data.pt')
    parser.add_argument("--pool_size", type=int, default=5)
    parser.add_argument("--short_selling", type=bool, default=False)
    parser.add_argument("--min_return", type=float, default=-0.3)
    parser.add_argument("--max_return", type=float, default=0.3)
    parser.add_argument("--num_plot_pts", type=int, default=10)
    parser.add_argument("--save_fig_name", type=str, default='./frontier.png')
    parser.add_argument("--save_fig_dpi", type=int, default=300)
    parser.add_argument("--save_pts_name", type=str, default='./frontier.csv')

    args = parser.parse_args()
    logging.info(f"========\t| =====")
    logging.info(f"Argument\t| Value")
    logging.info(f"--------\t| -----")
    for arg, value in vars(args).items():
        logging.info(f"{arg}\t| {value}")
    logging.info(f"========\t| =====")
    return args


def load_data(data_home, data_year, data_filename, pool_size):
    for root, folders, files in os.walk(data_home):
        assert len(folders) == 0, "Subfolder is not allowed."
        last_file = files[-1]
        df = pd.read_csv(os.path.join(data_home, last_file), index_col=1)
        stock_codes = df.index[:pool_size]
        pool = [
            Stock(data_home, stock_code, data_year)
            for stock_code in stock_codes
        ]
        pickle.dump(pool, open(data_filename, 'wb'))


def load_my_data(data_home, data_year, data_filename, stock_codes):
    pool = [
        Stock(data_home, stock_code, data_year) for stock_code in stock_codes
    ]
    pickle.dump(pool, open(data_filename, 'wb'))


def risk_minimize(stock_pool, short_selling, return_range, num_pts):
    def risk_fun(w: np.ndarray):
        """ risk function (to be minimized) """
        risk = 0
        for i in range(len(stock_pool)):
            for j in range(i, len(stock_pool)):
                risk += (2 * w[i] * w[j] * stock_pool[i].cov(stock_pool[j]))
        return risk

    returns = np.array([stock.mean for stock in stock_pool])

    def cons(return_required):
        """ optimization constraints """
        return ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        }, {
            'type': 'eq',
            'fun': lambda w: w @ returns - return_required
        })

    opt_args = {'fun': risk_fun, 'x0': np.zeros(len(stock_pool))}
    if not short_selling:
        opt_args.update({'bounds': opt.Bounds(0, 1)})

    frontier = list()
    for return_required in tqdm(np.linspace(*return_range, num_pts),
                                desc='Optimization'):
        opt_args.update({'constraints': cons(return_required)})
        opt_result = opt.minimize(**opt_args)

        opt_risk = np.sqrt(opt_result.fun)
        opt_return = opt_result.x @ returns
        frontier.append([opt_risk, opt_return])

    return frontier


def plot_frontier(frontier, save_fig_name, dpi):
    # check available styles, use:
    # ``` for style in plt.style.available: print(style) ```
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    ax.plot(*list(zip(*frontier)), 'o-')
    ax.set_xlabel('Risk (%)')
    ax.set_ylabel('Expected Return (%)')
    ax.set_xticklabels([f'{100*x:.3f}' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{100*y:.3f}' for y in ax.get_yticks()])
    ax.set_title('Efficient Frontier')
    fig.tight_layout()
    fig.savefig(save_fig_name, dpi=dpi)


def main():
    # get arguments
    args = get_args()

    # load data
    if not os.path.isfile(args.data_filename):
        logging.info(
            f"Data file '{args.data_filename}' doesn't exists. Loading data from {args.data_home}..."
        )
        load_data(
            args.data_home,
            args.data_year,
            args.data_filename,
            args.pool_size,
        )
    stock_pool = pickle.load(open(args.data_filename, 'rb'))

    # risk minimization
    frontier = risk_minimize(
        stock_pool,
        args.short_selling,
        [args.min_return, args.max_return],
        args.num_plot_pts,
    )

    # plot efficient frontier
    plot_frontier(
        frontier,
        args.save_fig_name,
        args.save_fig_dpi,
    )

    # save data points
    pd.DataFrame(frontier).to_csv(args.save_pts_name, header=None, index=None)


if __name__ == '__main__':
    main()