from __future__ import print_function
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class Stock(object):
    def __init__(self, data_home, code, year):
        self.data_home = data_home
        self.code = code
        self.returns = self.collect_data(year)
        self.len = len(self.returns)
        self.std = np.std(self.returns)
        self.mean = np.mean(self.returns)

    def collect_data(self, year):
        price = dict()
        for root, folders, files in os.walk(self.data_home):
            files = [file for file in files if file[:4] == str(year)]
            for file in tqdm(files, desc=self.code):
                file_path = os.path.join(self.data_home, file)
                file_name_wo_ext = os.path.splitext(file)[0]
                f = pd.read_csv(file_path, index_col=1)
                try:
                    price.update({file_name_wo_ext: f.loc[self.code, "close"]})
                except:
                    KeyError

        prices = pd.Series(price, dtype=np.float32)
        returns = np.diff(prices) / prices[:-1]
        assert len(
            returns) >= 200, f"{self.code} has only {len(returns)} entries."
        return returns

    def cov(self, another):
        return self.returns.cov(another.returns)

    def describe(self):
        print(f"Code\t{self.code}")
        print(f"Date\t{self.returns.index[0]} -- {self.returns.index[-1]}")
        print(f"Obs\t{self.len}")
        print(f"Mean\t{self.mean}")
        print(f"Sd\t{self.std}")
        print()
