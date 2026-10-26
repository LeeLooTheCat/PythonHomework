"""Find which weekday a 30-day month should start on to maximize average daily profit.

Usage:
    python bagel_month_start.py [bagelData.xlsx] [--model both|normal|gb]
                                [--price 2.5] [--cost 0.5] [--min-obs 3]

A 30-day month is 4 full weeks (28 days) plus 2 extra days. The 4 full weeks
contain every weekday exactly 4 times no matter where the month starts, so
the start day only decides which two consecutive weekdays occur a 5th time:
the start day itself and the day after it. The best start day is therefore
the one whose (start, next) pair has the highest combined expected profit.

Each weekday's expected profit comes from the optimized plan (best location
and bake quantity) produced by one or both of the earlier models:
    normal - bagel_optimizer.py (normal demand, sell-outs treated as censored)
    gb     - bagel_gradient_boosting.py (gradient boosting demand model)
"""

import argparse
import contextlib
import io

import pandas as pd

import bagel_gradient_boosting
import bagel_optimizer

DAY_ORDER = bagel_optimizer.DAY_ORDER
MONTH_DAYS = 30


def daily_plan(model, df, price, cost, min_obs):
    """Expected profit of the recommended plan for each weekday."""
    module = bagel_optimizer if model == "normal" else bagel_gradient_boosting
    with contextlib.redirect_stdout(io.StringIO()):  # hide the model's own report
        table = module.analyze(df, price, cost)
    picks = module.recommend(table, min_obs)
    picks["dayofweek"] = picks["dayofweek"].astype(str)
    return picks.set_index("dayofweek").loc[DAY_ORDER]


def month_scenarios(plan):
    """Total and average profit of a 30-day month for each possible start day."""
    profit = plan["exp_profit"]
    rows = []
    for i, start in enumerate(DAY_ORDER):
        days = [DAY_ORDER[(i + d) % 7] for d in range(MONTH_DAYS)]
        total = sum(profit[d] for d in days)
        extra = DAY_ORDER[i], DAY_ORDER[(i + 1) % 7]
        rows.append({
            "start_day": start,
            "extra_days": f"{extra[0][:3]} + {extra[1][:3]}",
            "extra_profit": profit[extra[0]] + profit[extra[1]],
            "month_profit": total,
            "avg_daily_profit": total / MONTH_DAYS,
        })
    return pd.DataFrame(rows).sort_values("avg_daily_profit", ascending=False)


def print_model_report(name, plan, scenarios):
    print(f"\n{name}")
    print("=" * 78)
    print("Expected profit per weekday with the optimized location and quantity:")
    for day, r in plan.iterrows():
        print(f"  {day:<10}{r['location']:<24}{r['best_qty']:>5} bagels   ${r['exp_profit']:>7.2f}")
    print()
    print(f"{'Start day':<12}{'5th-time days':<16}{'Extra-day $':>12}{'Month $':>12}{'Avg $/day':>12}")
    for rank, (_, r) in enumerate(scenarios.iterrows(), start=1):
        mark = "  <- best" if rank == 1 else ""
        print(f"{r['start_day']:<12}{r['extra_days']:<16}{r['extra_profit']:>12.2f}"
              f"{r['month_profit']:>12.2f}{r['avg_daily_profit']:>12.2f}{mark}")
    best, worst = scenarios.iloc[0], scenarios.iloc[-1]
    gain = best["month_profit"] - worst["month_profit"]
    print(f"\nBest start: {best['start_day']} (${best['avg_daily_profit']:.2f}/day). "
          f"That is ${gain:.2f} more per month than starting on {worst['start_day']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data", nargs="?", default="bagelData.xlsx", help="path to the Excel data")
    parser.add_argument("--model", choices=["both", "normal", "gb"], default="both")
    parser.add_argument("--price", type=float, default=2.5, help="profit per bagel sold before baking cost")
    parser.add_argument("--cost", type=float, default=0.5, help="cost to bake one bagel")
    parser.add_argument("--min-obs", type=int, default=3, help="minimum visits to trust a location")
    args = parser.parse_args()

    df = bagel_optimizer.load_data(args.data)
    models = {"normal": "Normal demand model (sell-outs treated as censored)",
              "gb": "Gradient boosting model"}
    chosen = list(models) if args.model == "both" else [args.model]

    print(f"A {MONTH_DAYS}-day month = 4 full weeks + 2 extra days. The start day and the "
          "day after it\noccur 5 times; every other weekday occurs 4 times.")
    for key in chosen:
        plan = daily_plan(key, df, args.price, args.cost, args.min_obs)
        print_model_report(models[key], plan, month_scenarios(plan))


if __name__ == "__main__":
    main()
