"""Pick the best cart location and how many bagels to make for each day of the week.

Usage:
    python bagel_optimizer.py [bagelData.xlsx] [--price 2.5] [--cost 0.5]
                              [--min-obs 3] [--out recommendations.csv]

How it works
------------
1. Economics. In the data, profit = 2.50 * bagels sold - 0.50 * bagels made
   (each bagel sold brings in $2.50, each bagel baked costs $0.50, leftovers
   are thrown away). The script checks this against the spreadsheet.

2. Demand. For every (day of week, location) pair, the bagels sold on each
   visit are a sample of demand. When a cart sold out (sold == made), the true
   demand was *at least* that many, so those visits are treated as censored.
   Demand is modeled as a normal distribution whose mean and standard
   deviation are estimated with an EM algorithm that accounts for sell-outs.

3. Quantity. This is a classic "newsvendor" problem: baking one bagel too few
   loses $2.00 of profit, baking one too many wastes $0.50. The expected
   profit is computed for every possible quantity and the best one is kept
   (it lands near the 80th percentile of demand).

4. Location. For each weekday the location with the highest expected profit
   wins. Pairs with fewer than --min-obs historical visits are listed but not
   recommended, since a handful of visits is not enough to trust.
"""

import argparse
import math
import sys
from statistics import NormalDist

import pandas as pd

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STD_NORMAL = NormalDist()


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


def fit_censored_normal(sold, censored, iterations=500):
    """Estimate demand mean/std when some observations are sell-outs (demand >= sold)."""
    mu = sum(sold) / len(sold)
    var = sum((x - mu) ** 2 for x in sold) / len(sold)
    sigma = max(math.sqrt(var), 1.0)
    if not any(censored):
        return mu, sigma

    for _ in range(iterations):
        e_x, e_x2 = [], []
        for x, is_cens in zip(sold, censored):
            if is_cens:
                # Moments of a normal truncated below at x.
                alpha = (x - mu) / sigma
                tail = max(1 - STD_NORMAL.cdf(alpha), 1e-12)
                lam = STD_NORMAL.pdf(alpha) / tail
                e_x.append(mu + sigma * lam)
                e_x2.append(mu ** 2 + sigma ** 2 + sigma * lam * (x + mu))
            else:
                e_x.append(x)
                e_x2.append(x ** 2)
        new_mu = sum(e_x) / len(e_x)
        new_sigma = max(math.sqrt(max(sum(e_x2) / len(e_x2) - new_mu ** 2, 0)), 1.0)
        if abs(new_mu - mu) < 1e-6 and abs(new_sigma - sigma) < 1e-6:
            break
        mu, sigma = new_mu, new_sigma
    return mu, sigma


def expected_sales(q, mu, sigma):
    """E[min(demand, q)] for normally distributed demand."""
    z = (q - mu) / sigma
    loss = STD_NORMAL.pdf(z) - z * (1 - STD_NORMAL.cdf(z))  # E[(Z - z)+]
    return mu - sigma * loss


def best_quantity(mu, sigma, price, cost):
    """Integer bake quantity that maximizes expected profit."""
    upper = int(math.ceil(mu + 5 * sigma))
    best = None
    for q in range(0, upper + 1):
        sales = expected_sales(q, mu, sigma)
        profit = price * sales - cost * q
        if best is None or profit > best[2]:
            best = (q, sales, profit)
    return best


def analyze(df, price, cost):
    rows = []
    for (day, loc), grp in df.groupby(["dayofweek", "location"]):
        sold = grp["bagelsold"].tolist()
        censored = (grp["bagelsold"] >= grp["bagelmade"]).tolist()
        mu, sigma = fit_censored_normal(sold, censored)
        q, sales, profit = best_quantity(mu, sigma, price, cost)
        rows.append({
            "dayofweek": day,
            "location": loc,
            "visits": len(grp),
            "sellouts": sum(censored),
            "avg_made": grp["bagelmade"].mean(),
            "avg_sold": grp["bagelsold"].mean(),
            "hist_avg_profit": (price * grp["bagelsold"] - cost * grp["bagelmade"]).mean(),
            "demand_mean": mu,
            "demand_std": sigma,
            "best_qty": q,
            "exp_sold": sales,
            "exp_profit": profit,
        })
    table = pd.DataFrame(rows)
    table["dayofweek"] = pd.Categorical(table["dayofweek"], DAY_ORDER, ordered=True)
    return table.sort_values(["dayofweek", "exp_profit"], ascending=[True, False])


def recommend(table, min_obs):
    picks = []
    for day, grp in table.groupby("dayofweek", observed=True):
        eligible = grp[grp["visits"] >= min_obs]
        if eligible.empty:
            eligible = grp  # fall back to whatever data exists
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
    print("Detail by day and location (best location first)")
    print("-" * 96)
    print(f"{'Day':<10}{'Location':<24}{'Visits':>7}{'Sellouts':>9}{'Hist $':>9}"
          f"{'Demand':>12}{'Bake':>7}{'Exp sold':>10}{'Exp $':>9}")
    for day, grp in table.groupby("dayofweek", observed=True):
        for _, r in grp.iterrows():
            flag = "" if r["visits"] >= min_obs else "  *"
            print(f"{day:<10}{r['location']:<24}{r['visits']:>7}{r['sellouts']:>9}"
                  f"{r['hist_avg_profit']:>9.2f}{r['demand_mean']:>7.0f} ±{r['demand_std']:<3.0f}"
                  f"{r['best_qty']:>7}{r['exp_sold']:>10.1f}{r['exp_profit']:>9.2f}{flag}")
        print()
    print(f"* fewer than {min_obs} visits: not used for the recommendation\n")

    print("RECOMMENDED PLAN")
    print("=" * 96)
    print(f"{'Day':<10}{'Location':<24}{'Bagels to make':>15}{'Expected profit':>17}"
          f"{'Next best location':>24}{'Edge':>6}")
    for _, r in picks.iterrows():
        print(f"{r['dayofweek']:<10}{r['location']:<24}{r['best_qty']:>15}"
              f"{'$' + format(r['exp_profit'], '.2f'):>17}{r['runner_up']:>24}"
              f"{'$' + format(r['edge_over_runner_up'], '.0f'):>6}")
    weekly = picks["exp_profit"].sum()
    print("-" * 96)
    print(f"Expected weekly profit following this plan: ${weekly:,.2f}")


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
