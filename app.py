from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request


app = Flask(__name__)
DATA_PATH = Path(__file__).with_name("ice_cream.csv")
MONTH_ORDER = ["April", "May", "June", "July", "August", "September", "October"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def month_order(data: pd.DataFrame) -> list[str]:
  return (
    data.sort_values("Date")
    .groupby("Month", sort=False)["Date"]
    .min()
    .sort_values()
    .index.tolist()
  )


def canonical_day_order(data: pd.DataFrame) -> list[str]:
  present_days = set(data["DayOfWeek"].dropna().unique())
  return [day for day in DAY_ORDER if day in present_days]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def build_design_matrix(
  frame: pd.DataFrame,
  month_levels: list[str],
  day_levels: list[str],
) -> np.ndarray:
  temperature = frame["Temperature"].to_numpy(dtype=float)
  rainfall = frame["Rainfall"].to_numpy(dtype=float)

  numeric = np.column_stack(
    [
      np.ones(len(frame)),
      temperature,
      rainfall,
      temperature * rainfall,
      temperature ** 2,
      rainfall ** 2,
    ]
  )

  month_dummies = pd.get_dummies(frame["Month"]).reindex(columns=month_levels[1:], fill_value=0)
  day_dummies = pd.get_dummies(frame["DayOfWeek"]).reindex(columns=day_levels[1:], fill_value=0)

  if month_dummies.empty:
    month_matrix = np.empty((len(frame), 0))
  else:
    month_matrix = month_dummies.to_numpy(dtype=float)

  if day_dummies.empty:
    day_matrix = np.empty((len(frame), 0))
  else:
    day_matrix = day_dummies.to_numpy(dtype=float)

  return np.column_stack([numeric, month_matrix, day_matrix])


def build_prediction_frame(
  temperature: float,
  rainfall: float,
  month: str,
  day_of_week: str,
) -> pd.DataFrame:
  return pd.DataFrame(
    {
      "Temperature": [float(temperature)],
      "Rainfall": [float(rainfall)],
      "Month": [month],
      "DayOfWeek": [day_of_week],
    }
  )


def fit_ridge_regression(design: np.ndarray, target: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def fit_regression(data: pd.DataFrame):
    month_levels = month_order(data)
    day_levels = canonical_day_order(data)
    ordered_data = data.sort_values("Date").reset_index(drop=True)
    ordered_target = ordered_data["IceCreamsSold"].to_numpy(dtype=float)
    ordered_design = build_design_matrix(ordered_data, month_levels, day_levels)

    split_index = max(1, int(len(ordered_data) * 0.8))
    train_design = ordered_design[:split_index]
    test_design = ordered_design[split_index:]
    train_target = ordered_target[:split_index]
    test_target = ordered_target[split_index:]

    validation_coefficients = fit_ridge_regression(train_design, train_target, alpha=1.0)

    train_predictions = train_design @ validation_coefficients
    train_residual_sum = float(np.sum((train_target - train_predictions) ** 2))
    train_total_sum = float(np.sum((train_target - train_target.mean()) ** 2))
    train_r_squared = 1.0 - train_residual_sum / train_total_sum if train_total_sum else 0.0

    if len(test_design):
        test_predictions = test_design @ validation_coefficients
        test_residual_sum = float(np.sum((test_target - test_predictions) ** 2))
        test_total_sum = float(np.sum((test_target - test_target.mean()) ** 2))
        r_squared = 1.0 - test_residual_sum / test_total_sum if test_total_sum else train_r_squared
        rmse = float(np.sqrt(test_residual_sum / len(test_design)))
    else:
        r_squared = train_r_squared
        rmse = float(np.sqrt(train_residual_sum / len(train_design)))

    coefficients = fit_ridge_regression(ordered_design, ordered_target, alpha=1.0)

    return {
        "month_levels": month_levels,
        "day_levels": day_levels,
        "weights": coefficients.tolist(),
        "intercept": float(coefficients[0]),
        "temp_coef": float(coefficients[1]),
        "rain_coef": float(coefficients[2]),
        "interaction_coef": float(coefficients[3]),
        "temp_squared_coef": float(coefficients[4]),
        "rain_squared_coef": float(coefficients[5]),
        "train_r_squared": train_r_squared,
        "r_squared": r_squared,
        "rmse": rmse,
    }


def build_context() -> dict:
    data = load_data()
    model = fit_regression(data)
    months = model["month_levels"]
    days = model["day_levels"]

    monthly = (
        data.groupby("Month", as_index=False)
        .agg(
            AverageSales=("IceCreamsSold", "mean"),
            TotalSales=("IceCreamsSold", "sum"),
            AverageTemperature=("Temperature", "mean"),
            AverageRainfall=("Rainfall", "mean"),
        )
        .set_index("Month")
        .reindex(months)
        .reset_index()
    )
    monthly = monthly.round({"AverageSales": 1, "TotalSales": 0, "AverageTemperature": 1, "AverageRainfall": 2})

    top_days = data.nlargest(5, "IceCreamsSold")[["Date", "DayOfWeek", "Temperature", "Rainfall", "IceCreamsSold"]].copy()
    top_days["Date"] = top_days["Date"].dt.strftime("%Y-%m-%d")

    hottest = data.loc[data["Temperature"].idxmax()]
    wettest = data.loc[data["Rainfall"].idxmax()]
    best_day = data.loc[data["IceCreamsSold"].idxmax()]

    average_temp = float(data["Temperature"].mean())
    average_rain = float(data["Rainfall"].mean())
    average_sales = float(data["IceCreamsSold"].mean())

    default_month = request.args.get("month", default=data["Month"].mode().iloc[0])
    if default_month not in months:
        default_month = months[0]

    default_day = request.args.get("day", default=data["DayOfWeek"].mode().iloc[0])
    if default_day not in days:
        default_day = days[0]

    default_temp = request.args.get("temp", type=float, default=average_temp)
    default_rain = request.args.get("rain", type=float, default=average_rain)
    prediction_frame = build_prediction_frame(default_temp, default_rain, default_month, default_day)
    predicted_features = build_design_matrix(prediction_frame, months, days)
    predicted_sales = (predicted_features @ np.array(model["weights"], dtype=float)).item()

    best_month_row = monthly.loc[monthly["AverageSales"].idxmax()]

    return {
        "rows": len(data),
        "average_sales": round(average_sales, 1),
        "total_sales": int(data["IceCreamsSold"].sum()),
        "average_temp": round(average_temp, 1),
        "average_rain": round(average_rain, 2),
        "best_day": {
            "date": best_day["Date"].strftime("%Y-%m-%d"),
            "sales": int(best_day["IceCreamsSold"]),
            "temperature": round(float(best_day["Temperature"]), 1),
            "rainfall": round(float(best_day["Rainfall"]), 2),
        },
        "hottest_day": {
            "date": hottest["Date"].strftime("%Y-%m-%d"),
            "temperature": round(float(hottest["Temperature"]), 1),
            "sales": int(hottest["IceCreamsSold"]),
        },
        "wettest_day": {
            "date": wettest["Date"].strftime("%Y-%m-%d"),
            "rainfall": round(float(wettest["Rainfall"]), 2),
            "sales": int(wettest["IceCreamsSold"]),
        },
        "monthly": monthly.to_dict(orient="records"),
        "top_days": top_days.to_dict(orient="records"),
        "best_month": {
            "month": str(best_month_row["Month"]),
            "sales": round(float(best_month_row["AverageSales"]), 1),
        },
        "model": model,
        "prediction_inputs": {
            "temp": round(float(default_temp), 1),
            "rain": round(float(default_rain), 2),
            "month": default_month,
            "day": default_day,
        },
        "month_options": months,
        "day_options": days,
        "predicted_sales": round(float(predicted_sales), 1),
    }


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ice Cream Shop Dashboard</title>
  <style>
    :root {
      --bg-1: #fff8f2;
      --bg-2: #ffe9ec;
      --bg-3: #f5fff8;
      --ink: #46362f;
      --muted: #7a665b;
      --card: rgba(255, 255, 255, 0.72);
      --line: rgba(110, 82, 68, 0.14);
      --accent: #f28e9c;
      --accent-2: #f7b267;
      --accent-3: #7cc4b8;
      --shadow: 0 24px 70px rgba(96, 54, 36, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(255, 219, 223, 0.85), transparent 32%),
        radial-gradient(circle at right center, rgba(255, 242, 200, 0.75), transparent 28%),
        linear-gradient(160deg, var(--bg-1), var(--bg-2) 48%, var(--bg-3));
      overflow-x: hidden;
    }

    body::before,
    body::after {
      content: "";
      position: fixed;
      border-radius: 999px;
      filter: blur(4px);
      pointer-events: none;
      z-index: 0;
      opacity: 0.7;
    }

    body::before {
      width: 220px;
      height: 220px;
      right: -70px;
      top: 70px;
      background: rgba(244, 169, 138, 0.24);
    }

    body::after {
      width: 280px;
      height: 280px;
      left: -100px;
      bottom: -80px;
      background: rgba(124, 196, 184, 0.18);
    }

    .shell {
      position: relative;
      z-index: 1;
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 18px 56px;
    }

    .hero {
      display: grid;
      gap: 18px;
      grid-template-columns: 1.4fr 0.9fr;
      align-items: stretch;
      margin-bottom: 18px;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 30px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
    }

    .hero-copy {
      padding: 34px;
      position: relative;
      overflow: hidden;
    }

    .hero-copy::after {
      content: "";
      position: absolute;
      right: -40px;
      top: -30px;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(247, 178, 103, 0.24), transparent 66%);
    }

    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(242, 142, 156, 0.12);
      color: #b45f72;
      font-size: 0.9rem;
      font-weight: 700;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2.2rem, 5vw, 4.2rem);
      line-height: 1.02;
      letter-spacing: -0.05em;
    }

    .lede {
      max-width: 56ch;
      margin: 16px 0 0;
      font-size: 1.05rem;
      line-height: 1.7;
      color: var(--muted);
    }

    .hero-side {
      display: grid;
      gap: 14px;
    }

    .stat-card {
      padding: 22px;
      display: grid;
      gap: 8px;
      align-content: start;
    }

    .stat-card small {
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .stat-card strong {
      font-size: 2rem;
      line-height: 1;
    }

    .stat-card span {
      color: var(--muted);
      line-height: 1.5;
    }

    .grid-3 {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      margin: 18px 0;
    }

    .metric {
      padding: 22px;
    }

    .metric .label {
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 10px;
      display: block;
    }

    .metric .value {
      font-size: 2.05rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      margin-bottom: 8px;
    }

    .metric .note {
      color: var(--muted);
      line-height: 1.5;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 18px 0 12px;
    }

    button,
    .button-link {
      border: 0;
      border-radius: 999px;
      padding: 13px 18px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
      color: #4b3229;
      background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.65));
      box-shadow: 0 10px 28px rgba(78, 48, 34, 0.08);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    button:hover,
    .button-link:hover {
      transform: translateY(-1px);
      box-shadow: 0 14px 34px rgba(78, 48, 34, 0.12);
    }

    .primary {
      background: linear-gradient(135deg, #f8b2bf, #f3a56e);
      color: white;
    }

    .section {
      padding: 24px;
      margin-top: 18px;
    }

    .section h2 {
      margin: 0 0 10px;
      font-size: 1.35rem;
    }

    .section p {
      margin: 0;
      color: var(--muted);
    }

    form.prediction {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
      align-items: end;
    }

    .field label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }

    .field input {
      width: 100%;
      border: 1px solid rgba(110, 82, 68, 0.15);
      border-radius: 18px;
      padding: 14px 16px;
      font: inherit;
      background: rgba(255, 255, 255, 0.76);
      color: var(--ink);
    }

    .field select {
      width: 100%;
      border: 1px solid rgba(110, 82, 68, 0.15);
      border-radius: 18px;
      padding: 14px 16px;
      font: inherit;
      background: rgba(255, 255, 255, 0.76);
      color: var(--ink);
    }

    .table-wrap {
      overflow-x: auto;
      margin-top: 14px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 700px;
    }

    th, td {
      text-align: left;
      padding: 14px 12px;
      border-bottom: 1px solid rgba(110, 82, 68, 0.12);
    }

    th {
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(53, 31, 22, 0.28);
      z-index: 20;
    }

    .modal.open { display: flex; }

    .modal-card {
      width: min(720px, 100%);
      background: rgba(255, 250, 247, 0.96);
      border-radius: 28px;
      box-shadow: 0 28px 90px rgba(44, 25, 18, 0.28);
      border: 1px solid rgba(110, 82, 68, 0.12);
      padding: 26px;
      max-height: 88vh;
      overflow: auto;
    }

    .modal-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 14px;
    }

    .close {
      width: 44px;
      height: 44px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: rgba(242, 142, 156, 0.14);
      color: #9e5565;
      font-size: 1.3rem;
      font-weight: 800;
    }

    .mini-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-top: 18px;
    }

    .mini {
      background: rgba(255, 255, 255, 0.8);
      border-radius: 20px;
      border: 1px solid rgba(110, 82, 68, 0.1);
      padding: 16px;
    }

    .mini strong {
      display: block;
      font-size: 1.2rem;
      margin-bottom: 6px;
    }

    .tiny {
      color: var(--muted);
      line-height: 1.55;
    }

    .footer-note {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.95rem;
    }

    @media (max-width: 920px) {
      .hero,
      .grid-3,
      form.prediction,
      .mini-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 640px) {
      .shell { padding: 18px 12px 42px; }
      .hero-copy, .section, .stat-card, .metric, .modal-card { padding: 18px; }
      h1 { font-size: 2.1rem; }
      .actions { gap: 10px; }
      button, .button-link { width: 100%; text-align: center; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="panel hero-copy">
        <div class="eyebrow">Soft shop dashboard</div>
        <h1>Ice Cream Shop Sales Dashboard</h1>
        <p class="lede">
          A small Flask app for Render with a simple linear regression model,
          monthly sales summaries, and gentle popup views for sales, temperature,
          and rainfall. The look is intentionally soft and cute for a cozy shop feel.
        </p>
        <div class="actions">
          <button class="primary" type="button" onclick="openModal('salesModal')">Sales Popup</button>
          <button type="button" onclick="openModal('tempModal')">Temperature Popup</button>
          <button type="button" onclick="openModal('rainModal')">Rainfall Popup</button>
        </div>
      </div>

      <div class="hero-side">
        <div class="panel stat-card">
          <small>Regression</small>
          <strong>{{ model.r_squared | round(3) }}</strong>
          <span>Holdout score from the enhanced weather model.</span>
        </div>
        <div class="panel stat-card">
          <small>Model error</small>
          <strong>{{ model.rmse | round(1) }}</strong>
          <span>Root mean squared error in sold scoops.</span>
        </div>
      </div>
    </section>

    <section class="grid-3">
      <div class="panel metric">
        <span class="label">Average sales</span>
        <div class="value">{{ average_sales }}</div>
        <div class="note">Typical daily sales across the uploaded data.</div>
      </div>
      <div class="panel metric">
        <span class="label">Best month</span>
        <div class="value">{{ best_month.month }}</div>
        <div class="note">Highest average monthly sales at {{ best_month.sales | round(1) }}.</div>
      </div>
      <div class="panel metric">
        <span class="label">Top day</span>
        <div class="value">{{ best_day.sales }}</div>
        <div class="note">Sold on {{ best_day.date }} when it was {{ best_day.temperature }} degrees.</div>
      </div>
    </section>

    <section class="panel section">
      <h2>Quick prediction</h2>
      <p>Enter weather values and optional season details for a more accurate sales prediction.</p>
      <form class="prediction" method="get">
        <div class="field">
          <label for="temp">Temperature</label>
          <input id="temp" name="temp" type="number" step="0.1" value="{{ prediction_inputs.temp }}">
        </div>
        <div class="field">
          <label for="rain">Rainfall</label>
          <input id="rain" name="rain" type="number" step="0.01" value="{{ prediction_inputs.rain }}">
        </div>
        <div class="field">
          <label for="month">Month</label>
          <select id="month" name="month">
            {% for month in month_options %}
            <option value="{{ month }}" {% if month == prediction_inputs.month %}selected{% endif %}>{{ month }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="day">Day of week</label>
          <select id="day" name="day">
            {% for day in day_options %}
            <option value="{{ day }}" {% if day == prediction_inputs.day %}selected{% endif %}>{{ day }}</option>
            {% endfor %}
          </select>
        </div>
        <button class="primary" type="submit">Predict sales</button>
      </form>
      <div class="footer-note">Predicted sales: <strong>{{ predicted_sales }}</strong> cups.</div>
      <div class="footer-note">
        The model uses temperature, rainfall, month, and weekday context with ridge regression.
      </div>
    </section>

    <section class="panel section">
      <h2>Monthly sales snapshot</h2>
      <p>Average sales, total sales, and weather averages for each month in the file.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Month</th>
              <th>Average sales</th>
              <th>Total sales</th>
              <th>Avg temperature</th>
              <th>Avg rainfall</th>
            </tr>
          </thead>
          <tbody>
            {% for row in monthly %}
            <tr>
              <td>{{ row.Month }}</td>
              <td>{{ row.AverageSales }}</td>
              <td>{{ row.TotalSales }}</td>
              <td>{{ row.AverageTemperature }}</td>
              <td>{{ row.AverageRainfall }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <div class="modal" id="salesModal" onclick="closeIfBackdrop(event, 'salesModal')">
    <div class="modal-card">
      <div class="modal-head">
        <div>
          <h2>Sales popup</h2>
          <p>What the sales numbers look like in simple, easy-to-read form.</p>
        </div>
        <button class="close" type="button" onclick="closeModal('salesModal')">&times;</button>
      </div>
      <div class="mini-grid">
        <div class="mini"><strong>{{ total_sales }}</strong><span class="tiny">Total scoops sold.</span></div>
        <div class="mini"><strong>{{ average_sales }}</strong><span class="tiny">Average sales per day.</span></div>
        <div class="mini"><strong>{{ best_day.date }}</strong><span class="tiny">Best day for sales.</span></div>
        <div class="mini"><strong>{{ best_month.month }}</strong><span class="tiny">Best month by average sales.</span></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Day</th>
              <th>Temp</th>
              <th>Rain</th>
              <th>Sales</th>
            </tr>
          </thead>
          <tbody>
            {% for row in top_days %}
            <tr>
              <td>{{ row.Date }}</td>
              <td>{{ row.DayOfWeek }}</td>
              <td>{{ row.Temperature }}</td>
              <td>{{ row.Rainfall }}</td>
              <td>{{ row.IceCreamsSold }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="modal" id="tempModal" onclick="closeIfBackdrop(event, 'tempModal')">
    <div class="modal-card">
      <div class="modal-head">
        <div>
          <h2>Temperature popup</h2>
          <p>Warm days usually lift sales. This popup shows the temperature side of the story.</p>
        </div>
        <button class="close" type="button" onclick="closeModal('tempModal')">&times;</button>
      </div>
      <div class="mini-grid">
        <div class="mini"><strong>{{ average_temp }}</strong><span class="tiny">Average temperature in the file.</span></div>
        <div class="mini"><strong>{{ hottest_day.temperature }}</strong><span class="tiny">Warmest day recorded.</span></div>
        <div class="mini"><strong>{{ hottest_day.date }}</strong><span class="tiny">Warmest date.</span></div>
        <div class="mini"><strong>{{ model.temp_coef | round(2) }}</strong><span class="tiny">Temperature effect in the model.</span></div>
      </div>
    </div>
  </div>

  <div class="modal" id="rainModal" onclick="closeIfBackdrop(event, 'rainModal')">
    <div class="modal-card">
      <div class="modal-head">
        <div>
          <h2>Rainfall popup</h2>
          <p>Rain tends to cool demand. This popup keeps the rainfall numbers simple.</p>
        </div>
        <button class="close" type="button" onclick="closeModal('rainModal')">&times;</button>
      </div>
      <div class="mini-grid">
        <div class="mini"><strong>{{ average_rain }}</strong><span class="tiny">Average rainfall in the file.</span></div>
        <div class="mini"><strong>{{ wettest_day.rainfall }}</strong><span class="tiny">Wettest day recorded.</span></div>
        <div class="mini"><strong>{{ wettest_day.date }}</strong><span class="tiny">Wettest date.</span></div>
        <div class="mini"><strong>{{ model.rain_coef | round(2) }}</strong><span class="tiny">Rainfall effect in the model.</span></div>
      </div>
    </div>
  </div>

  <script>
    function openModal(id) {
      document.getElementById(id).classList.add("open");
    }

    function closeModal(id) {
      document.getElementById(id).classList.remove("open");
    }

    function closeIfBackdrop(event, id) {
      if (event.target && event.target.id === id) {
        closeModal(id);
      }
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    context = build_context()
    return render_template_string(PAGE, **context)


if __name__ == "__main__":
    app.run(debug=True)