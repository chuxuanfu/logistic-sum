# Chuxuan Fu Logistic Sum function

Plot multiple logistic functions and their summed curve.

Each function uses:

```text
f(x) = L / (1 + e^(-k * (x - x0)))
```

Edit the `PARAMETERS` section in `plot_sigmoid_sum.py` to control `L`, `x0`,
the shared `K`, and the x-axis range.

## Run

```bash
python3.12 -m venv .venv312
.venv312/bin/pip install -r requirements.txt
.venv312/bin/python3.12 plot_sigmoid_sum.py
```

The script writes `sigmoid_sum.png` by default.
