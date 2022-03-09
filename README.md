# Portfolio

> [Modern portfolio theory - Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

## Efficient Frontier

### Model

<!-- $$
\begin{split}
\min_{\boldsymbol{w}} ~~& \sigma_P^2 = \sum_i\sum_j w_i w_j \sigma_i\sigma_j \\
\text{s.t.}           ~~& \boldsymbol{w}^\top \boldsymbol{1} = 1             \\
                      ~~& \boldsymbol{w}^\top \boldsymbol{\mu} = \mu_P       \\
\end{split}
$$ -->

<img src="https://latex.codecogs.com/svg.image?\min_{\boldsymbol{w}}&space;~~&space;\sigma_P^2&space;=&space;\sum_i\sum_j&space;w_i&space;w_j&space;\sigma_i\sigma_j&space;\\~~~~~~~\text{s.t.}&space;~~\boldsymbol{w}^\top&space;\boldsymbol{1}&space;=&space;1&space;\\~~~~~~~~~~~~~\boldsymbol{w}^\top&space;\boldsymbol{\mu}&space;=&space;\mu_P&space;" title="https://latex.codecogs.com/svg.image?\min_{\boldsymbol{w}} ~~ \sigma_P^2 = \sum_i\sum_j w_i w_j \sigma_i\sigma_j \\~~~~~~~\text{s.t.} ~~\boldsymbol{w}^\top \boldsymbol{1} = 1 \\~~~~~~~~~~~~~\boldsymbol{w}^\top \boldsymbol{\mu} = \mu_P " />

### Usage

```python
# you can specify your favorite arguments
bash frontier.sh
# or use default arguments
python frontier.py
```

### Result

![frontier](./frontier.png)

## Asset Allocation

### Model

<!-- $$
\begin{split}
\max_{\mu,\sigma} ~~& U = \mu-\frac12 A\sigma^2 \\
\text{s.t.} ~~& \mu=r_F + \frac{\sigma}{\sigma_P} (\mu_P-r_F) \\
\end{split}
$$ -->

<img src="https://latex.codecogs.com/svg.image?\max_{\mu,\sigma}&space;~~&space;U&space;=&space;\mu-\frac12&space;A\sigma^2&space;\\~~~~~~~~\text{s.t.}&space;~~&space;\mu=r_F&space;&plus;&space;\frac{\sigma}{\sigma_P}&space;(\mu_P-r_F)&space;" title="https://latex.codecogs.com/svg.image?\max_{\mu,\sigma} ~~ U = \mu-\frac12 A\sigma^2 \\~~~~~~~~\text{s.t.} ~~ \mu=r_F + \frac{\sigma}{\sigma_P} (\mu_P-r_F) " />

> Subscript clarification:
>
> - F is for "risk-**f**ree"
> - P is for "**p**ortfolio"

### Usage

```bash
python allocation.py
```

### Result

![allocation](./allocation.png)

## Data

There is only one sample csv file [`2011-01-31.csv`](./data/2011-01-31.csv) in `./data/`. Collect other data by your self.
