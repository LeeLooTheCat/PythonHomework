"""Choose the best cart location and bake quantity per weekday with gradient boosting.

Usage:
    python bagel_gradient_boosting.py [bagelData.xlsx] [--price 2.5] [--cost 0.5]
                                      [--min-obs 3] [--out plan_gb.csv]

How it works
------------
1. Economics. In the data, profit = 2.50 * bagels sold - 0.50 * bagels made.
   Baking one bagel too few loses $2.00; baking one too many wastes $0.50.

2. Demand model. Gradient boosting regressors (scikit-learn) learn bagel
   demand from the day of week and the location (one-hot encoded; trees of
   depth 2 capture day x location interactions):
     * one model with squared-error loss predicts average demand, and
     * 20 models with quantile loss (5%, 10%, ... 95% levels) predict the
       whole demand distribution, i.e. how demand varies from visit to visit.

3. Quantity. For every possible bake quantity Q, expected sales are averaged
   over the predicted demand quantiles, giving expected profit
   price * E[min(demand, Q)] - cost * Q. The Q with the highest expected
   profit is kept (it lands near the 80% demand quantile).

4. Location. For each weekday the location with the highest expected profit
   wins. Pairs with fewer than --min-obs historical visits are shown but not
   recommended, because the model has little evidence there.

Note: on days the cart sold out, real demand was higher than bagels sold, so
the models slightly understate demand at the busiest spots.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
QUANTILES = (np.arange(20) + 0.5) / 20  # 0.025, 0.075, ..., 0.975 -> 20 equal-weight points
GB_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=2,
                 min_samples_leaf=3, subsample=0.8, random_state=0)


def load_data(path):
    df = pd.read_excel(path)
    required = {"dayofweek", "location", "bagelmade", "bagelsold"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing column(s) in {path}: {', '.join(sorted(missing))}")
    df = df.dropna(subset=list(required))
    df["dayofweek"] = df["dayofweek"].str.strip().str.title()
    df["location"] = df["location"].str.strip()
    return df


def check_economics(df, price, cost):
    """Report how well profit = price*sold - cost*made matches the data."""
    if "profit" not in df.columns:
        return
    predicted = price * df["bagelsold"] - cost * df["bagelmade"]
    max_err = (predicted - df["profit"]).abs().max()
    status = "matches" if max_err < 0.01 else f"is off by up to ${max_err:.2f}"
    print(f"Profit model: ${price:.2f} per bagel sold - ${cost:.2f} per bagel made "
          f"({status} the spreadsheet)\n")


def encode(frame, columns):
    """One-hot encode day and location, aligned to the training columns."""
    X = pd.get_dummies(frame[["dayofweek", "location"]], dtype=float)
    return X.reindex(columns=columns, fill_value=0.0)


def cross_validate(df, columns, folds=5):
    """Compare out-of-sample error of the boosted model with a simple average."""
    X = encode(df, columns).to_numpy()
    y = df["bagelsold"].to_numpy()
    gb_err, base_err = [], []
    for train, test in KFold(folds, shuffle=True, random_state=0).split(X):
        model = GradientBoostingRegressor(**GB_PARAMS).fit(X[train], y[train])
        gb_err.append(np.abs(model.predict(X[test]) - y[test]).mean())

        train_df = df.iloc[train]
        cell_mean = train_df.groupby(["dayofweek", "location"])["bagelsold"].mean()
        keys = list(zip(df.iloc[test]["dayofweek"], df.iloc[test]["location"]))
        baseline = np.array([cell_mean.get(k, y[train].mean()) for k in keys])
        base_err.append(np.abs(baseline - y[test]).mean())
    print(f"{folds}-fold cross-validation, mean absolute error in bagels sold:")
    print(f"  gradient boosting       {np.mean(gb_err):6.2f}")
    print(f"  day+location averages   {np.mean(base_err):6.2f}\n")


def fit_models(df, columns):
    X = encode(df, columns)
    y = df["bagelsold"]
    mean_model = GradientBoostingRegressor(loss="squared_error", **GB_PARAMS).fit(X, y)
    quantile_models = [
        GradientBoostingRegressor(loss="quantile", alpha=a, **GB_PARAMS).fit(X, y)
        for a in QUANTILES
    ]
    return mean_model, quantile_models


def best_quantity(demand_points, price, cost):
    """Integer bake quantity maximizing expected profit over sampled demand."""
    quantities = np.arange(0, int(np.ceil(demand_points.max())) + 2)
    exp_sold = np.minimum(demand_points[None, :], quantities[:, None]).mean(axis=1)
    exp_profit = price * exp_sold - cost * quantities
    i = int(exp_profit.argmax())
    return int(quantities[i]), exp_sold[i], exp_profit[i]


def analyze(df, price, cost):
    columns = pd.get_dummies(df[["dayofweek", "location"]], dtype=float).columns
    cross_validate(df, columns)
    mean_model, quantile_models = fit_models(df, columns)

    grid = pd.MultiIndex.from_product(
        [DAY_ORDER, sorted(df["location"].unique())], names=["dayofweek", "location"]
    ).to_frame(index=False)
    X = encode(grid, columns)
    mean_demand = mean_model.predict(X)
    # Sort so quantiles never cross (each model is fitted independently).
    demand_quantiles = np.sort(np.column_stack([m.predict(X) for m in quantile_models]), axis=1)

    visits = df.groupby(["dayofweek", "location"]).size()
    rows = []
    for i, r in grid.iterrows():
        q, sold, profit = best_quantity(demand_quantiles[i], price, cost)
        rows.append({
            "dayofweek": r["dayofweek"],
            "location": r["location"],
            "visits": int(visits.get((r["dayofweek"], r["location"]), 0)),
            "pred_demand": mean_demand[i],
            "demand_p10": np.interp(0.10, QUANTILES, demand_quantiles[i]),
            "demand_p90": np.interp(0.90, QUANTILES, demand_quantiles[i]),
            "best_qty": q,
            "exp_sold": sold,
            "exp_profit": profit,
        })
    table = pd.DataFrame(rows)
    table["dayofweek"] = pd.Categorical(table["dayofweek"], DAY_ORDER, ordered=True)
    return table.sort_values(["dayofweek", "exp_profit"], ascending=[True, False])


def recommend(table, min_obs):
    picks = []
    for _, grp in table.groupby("dayofweek", observed=True):
        eligible = grp[grp["visits"] >= min_obs]
        if eligible.empty:
            eligible = grp[grp["visits"] > 0]
        ranked = eligible.sort_values("exp_profit", ascending=False)
        best = ranked.iloc[0].to_dict()
        runner = ranked.iloc[1] if len(ranked) > 1 else None
        best["runner_up"] = runner["location"] if runner is not None else ""
        best["edge_over_runner_up"] = (
            best["exp_profit"] - runner["exp_profit"] if runner is not None else float("nan")
        )
        picks.append(best)
    return pd.DataFrame(picks)


def print_report(table, picks, min_obs):
    print("Gradient boosting predictions by day and location (best first)")
    print("-" * 92)
    print(f"{'Day':<10}{'Location':<24}{'Visits':>7}{'Demand':>8}{'10%-90% range':>16}"
          f"{'Bake':>7}{'Exp sold':>10}{'Exp $':>9}")
    for day, grp in table.groupby("dayofweek", observed=True):
        for _, r in grp.iterrows():
            flag = "" if r["visits"] >= min_obs else "  *"
            band = f"{r['demand_p10']:.0f}-{r['demand_p90']:.0f}"
            print(f"{day:<10}{r['location']:<24}{r['visits']:>7}{r['pred_demand']:>8.0f}"
                  f"{band:>16}{r['best_qty']:>7}{r['exp_sold']:>10.1f}{r['exp_profit']:>9.2f}{flag}")
        print()
    print(f"* fewer than {min_obs} visits: not used for the recommendation\n")

    print("RECOMMENDED PLAN (gradient boosting)")
    print("=" * 92)
    print(f"{'Day':<10}{'Location':<24}{'Bagels to make':>15}{'Expected profit':>17}"
          f"{'Next best location':>24}")
    for _, r in picks.iterrows():
        print(f"{r['dayofweek']:<10}{r['location']:<24}{r['best_qty']:>15}"
              f"{'$' + format(r['exp_profit'], '.2f'):>17}{r['runner_up']:>24}")
    print("-" * 92)
    print(f"Expected weekly profit following this plan: ${picks['exp_profit'].sum():,.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", nargs="?", default="bagelData.xlsx", help="path to the Excel data")
    parser.add_argument("--price", type=float, default=2.5, help="profit per bagel sold before baking cost")
    parser.add_argument("--cost", type=float, default=0.5, help="cost to bake one bagel")
    parser.add_argument("--min-obs", type=int, default=3, help="minimum visits to trust a location")
    parser.add_argument("--out", help="optional CSV file for the recommended plan")
    args = parser.parse_args()

    df = load_data(args.data)
    check_economics(df, args.price, args.cost)
    table = analyze(df, args.price, args.cost)
    picks = recommend(table, args.min_obs)
    print_report(table, picks, args.min_obs)

    if args.out:
        picks[["dayofweek", "location", "best_qty", "exp_sold", "exp_profit",
               "visits", "runner_up"]].round(2).to_csv(args.out, index=False)
        print(f"\nPlan saved to {args.out}")


if __name__ == "__main__":
    main()
