#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:14:12 2026

@author: debasmitasen
"""
# Business requirement: Demand Forecast products per month.
# Compare model performance using A/B testing.

import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# We read the files and perform  and outliers.
inv=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_inventoryNew.csv")
orditems=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_order_items.csv")
orders=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_orders.csv")
products=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_products.csv")

#Perform EDA to study the data and check for null values

pd.set_option('display.max_columns', None)
inv.shape
inv.describe()
inv.head(5)
inv.isnull().sum()


orditems.shape
orditems.describe()
orditems.head(5)
orditems.isnull().sum()


orders.shape
orders.describe()
orders.head(5)
orders.isnull().sum()


products.shape
products.describe()
products.head(5)
products.isnull().sum()

#Check outliers visually. If needed we can dig down to te specific rows that have outlier values.

inv.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
orditems.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
orders.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
products.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
#No outliers detected in all the datasets.

# Prepare the data by joining the tables and then study the joined data


df = orditems.merge(orders[['order_id','order_date','store_id']], on='order_id')
df = df.merge(products[['product_id','product_name','category']], on='product_id')
df.head(5)
df['order_date'] = pd.to_datetime(df['order_date'])
print(df['order_date'].dt.to_period('M').nunique(), "months")#21 months
print(df['store_id'].nunique(), "stores")#5000 stores
print(df['product_id'].nunique(), "products")#268 products
print(len(df), "total rows")#5000 total rows

# Build monthly time series for the different products

df['month'] = df['order_date'].dt.to_period('M')

# Group by both product_id and month
ts = (
    df.groupby(['product_id', 'month'])['quantity']
    .sum()
    .reset_index()
)

# Fill missing months with 0 so every product has a complete calendar

all_months   = ts['month'].unique()
all_products = ts['product_id'].unique()

full_index = pd.MultiIndex.from_product(
    [all_products, all_months], names=['product_id', 'month']
)
ts = (
    ts.set_index(['product_id', 'month'])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

ts['month'] = ts['month'].dt.to_timestamp()   # Prophet needs Timestamps

print(f"\nTime-series shape: {ts.shape}")
print(f"Months per product: {ts.groupby('product_id')['month'].count().describe()}")

# Since this is a time series data and can have seasonality trends as we are dealing with consumer products, we use Prophet as model A
MIN_MONTHS     = 6      # minimum history required
FORECAST_MONTHS = 3     # how many months ahead to predict

results = []
skipped = []

for product, group in ts.groupby('product_id'):
    temp = (
        group[['month', 'quantity']]
        .rename(columns={'month': 'ds', 'quantity': 'y'})
        .sort_values('ds')
    )

    # Quality checks
    if len(temp) < MIN_MONTHS:
        skipped.append((product, 'too few months'))
        continue
    if temp['y'].isna().any():
        skipped.append((product, 'NaN values'))
        continue
    if temp['ds'].duplicated().any():
        skipped.append((product, 'duplicate dates'))
        continue
    if temp['y'].sum() == 0:
        skipped.append((product, 'all zeros'))
        continue

    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,   # monthly data → no weekly signal
            daily_seasonality=False,
            seasonality_mode='multiplicative',  # better for retail demand
        )
        model.fit(temp)

        future   = model.make_future_dataframe(periods=FORECAST_MONTHS, freq='MS')
        forecast = model.predict(future)

        # Clip negatives — demand can't be < 0
        forecast['yhat']       = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

        forecast['product_id'] = product
        results.append(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product_id']])

    except Exception as e:
        skipped.append((product, str(e)))

print(f"\n Forecasted : {len(results)} products")
print(f" Skipped    : {len(skipped)} products")

# Results for ModelA-Prophet

predictions = pd.concat(results, ignore_index=True)

# Separate historical fit vs future forecast
last_date    = ts['month'].max()
future_preds = predictions[predictions['ds'] > last_date].copy()

# Merge product names back
future_preds = future_preds.merge(
    products[['product_id', 'product_name', 'category']], on='product_id'
)

print("\nForecast preview:")
print(future_preds.sort_values(['product_id', 'ds']).head(15))

# Visualizing top 6 products

top_products = (
    ts.groupby('product_id')['quantity'].sum()
    .nlargest(6).index
)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for ax, product in zip(axes, top_products):
    hist = ts[ts['product_id'] == product]
    pred = predictions[predictions['product_id'] == product]
    name = products.loc[products['product_id'] == product, 'product_name'].values[0]

    ax.plot(hist['month'], hist['quantity'], 'o-', label='Actual', color='steelblue')
    ax.plot(pred['ds'],    pred['yhat'],    '--',  label='Forecast', color='darkorange')
    ax.fill_between(pred['ds'], pred['yhat_lower'], pred['yhat_upper'],
                    alpha=0.2, color='darkorange')
    ax.axvline(last_date, color='gray', linestyle=':', label='Forecast start')
    ax.set_title(name[:30], fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Demand Forecast — Top 6 Products", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

#Evaluation of Prophet

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Merge actuals with Prophet predictions
# predictions already has: ds, yhat, yhat_lower, yhat_upper, product_id

# Keep only in-sample / test period rows (dates we have actuals for)
actuals = ts[['product_id', 'month', 'quantity']].rename(columns={'month': 'ds'})

eval_df = predictions.merge(actuals, on=['product_id', 'ds'], how='inner')
eval_df['yhat'] = eval_df['yhat'].clip(lower=0)

print(f"Evaluating on {eval_df['ds'].nunique()} months, {eval_df['product_id'].nunique()} products\n")

# Accuracy metrics
mae  = mean_absolute_error(eval_df['quantity'], eval_df['yhat'])
rmse = np.sqrt(mean_squared_error(eval_df['quantity'], eval_df['yhat']))

mask = eval_df['quantity'] > 0
mape = (
    np.abs((eval_df.loc[mask, 'quantity'] - eval_df.loc[mask, 'yhat'])
           / eval_df.loc[mask, 'quantity'])
).mean() * 100

# Bias: positive = over-forecasting, negative = under-forecasting
bias = (eval_df['yhat'] - eval_df['quantity']).mean()

print("── Overall ──────────────────────────────")
print(f"  MAE  : {mae:.2f}   (avg units off per month)")
print(f"  RMSE : {rmse:.2f}   (penalises large errors more)")
print(f"  MAPE : {mape:.1f}%  (% error, non-zero actuals only)")
print(f"  Bias : {bias:+.2f}  ({'over' if bias > 0 else 'under'}-forecasting on average)")

# Per product accuracy

def product_metrics(g):
    mask = g['quantity'] > 0
    mape = (
        np.abs((g.loc[mask, 'quantity'] - g.loc[mask, 'yhat']) / g.loc[mask, 'quantity'])
    ).mean() * 100 if mask.any() else np.nan

    return pd.Series({
        'mae':         mean_absolute_error(g['quantity'], g['yhat']),
        'rmse':        np.sqrt(mean_squared_error(g['quantity'], g['yhat'])),
        'mape':        mape,
        'bias':        (g['yhat'] - g['quantity']).mean(),
        'n_months':    len(g),
    })

per_product = (
    eval_df.groupby('product_id')
    .apply(product_metrics)
    .sort_values('mape')
    .reset_index()
)

# Merge product names
per_product = per_product.merge(
    products[['product_id', 'product_name', 'category']], on='product_id'
)

print("\n── Best 10 products (lowest MAPE) ───────")
print(per_product[['product_name', 'mae', 'rmse', 'mape', 'bias', 'n_months']].head(10).to_string(index=False))

print("\n── Worst 10 products (highest MAPE) ─────")
print(per_product[['product_name', 'mae', 'rmse', 'mape', 'bias', 'n_months']].tail(10).to_string(index=False))

#Accuracy buckets for products
bins   = [0, 10, 20, 30, 50, float('inf')]
labels = ['<10%', '10–20%', '20–30%', '30–50%', '>50%']

per_product['mape_bucket'] = pd.cut(per_product['mape'], bins=bins, labels=labels)
bucket_counts = per_product['mape_bucket'].value_counts().sort_index()

print("\n── MAPE distribution across products ────")
for bucket, count in bucket_counts.items():
    pct = count / len(per_product) * 100
    bar = '█' * int(pct / 2)
    print(f"  {bucket:8s} {bar:25s} {count:3d} products ({pct:.0f}%)")

# Visualizing accuracy
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Actual vs predicted scatter
axes[0].scatter(eval_df['quantity'], eval_df['yhat'], alpha=0.3, s=10, color='steelblue')
lim = max(eval_df['quantity'].max(), eval_df['yhat'].max())
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# MAPE distribution histogram
axes[1].hist(per_product['mape'].dropna(), bins=20, color='steelblue', edgecolor='white')
axes[1].axvline(mape, color='red', linestyle='--', label=f'Mean MAPE {mape:.1f}%')
axes[1].set_xlabel('MAPE (%)')
axes[1].set_ylabel('Products')
axes[1].set_title('MAPE distribution')
axes[1].legend()

# Residuals over time
monthly_mae = (
    eval_df.groupby('ds')
    .apply(lambda g: mean_absolute_error(g['quantity'], g['yhat']))
    .reset_index(name='mae')
)
axes[2].plot(monthly_mae['ds'], monthly_mae['mae'], 'o-', color='steelblue')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('MAE')
axes[2].set_title('MAE over time')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle('Prophet — Forecast Accuracy', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

#Interpretation: Predictions closely match actuals- points very close to the red dashed line, Average error (MAPE) is extremely low-data is extremely stable
# Monthly MAE is stable with only mild fluctuations-no sign of seasonal peroids where model fails or increasing error

# Model B: Sarimax

#imort dependencies
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

eval_rows = []          # <-- evaluation results stored here
all_forecasts = []      # <-- future forecasts stored here

forecast_horizon = 6    # example: forecast next 6 months

last_month = ts['month'].max()
future_months = pd.date_range(
    start=last_month + pd.offsets.MonthBegin(1),
    periods=forecast_horizon,
    freq='MS'
)

for pid in tqdm(ts['product_id'].unique()):
    
    df_prod = ts[ts['product_id'] == pid].copy()
    df_prod = df_prod.sort_values('month').set_index('month')
    y = df_prod['quantity']
    
    try:

        model = sm.tsa.statespace.SARIMAX(
            y,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)


        pred_in_sample = results.get_prediction(
            start=y.index[0],
            end=y.index[-1]
        ).predicted_mean

        mae  = mean_absolute_error(y, pred_in_sample)
        mape = mean_absolute_percentage_error(y, pred_in_sample)
        rmse = np.sqrt(mean_squared_error(y, pred_in_sample))

        eval_rows.append([pid, mae, mape, rmse])


        forecast = results.get_forecast(steps=forecast_horizon)
        pred_future = forecast.predicted_mean

        for m, val in zip(future_months, pred_future):
            all_forecasts.append([pid, m, max(val, 0)])

    except Exception as e:
        # If model fails, fill with zeros
        eval_rows.append([pid, None, None, None])
        
        sarimax_eval = pd.DataFrame(
    eval_rows,
    columns=['product_id','mae','mape','rmse']
)

sarimax_forecasts = pd.DataFrame(
    all_forecasts,
    columns=['product_id','month','forecast_quantity']
)
# Evaluation summary statistics
sarimax_eval.describe()
#MAPE distribution
sarimax_eval['mape'].hist(bins=30, figsize=(8,4))
#slightly skewed distribution

# Best/ worst products
sarimax_eval.sort_values('mape').head(10)
sarimax_eval.sort_values('mape').tail(10)

#Visualizing the evaluation of Sarimax

import matplotlib.pyplot as plt

# Evaluation

# Build a combined dataframe of actual vs predicted for all products
actual_pred_list = []

for pid in ts['product_id'].unique():
    df_prod = ts[ts['product_id'] == pid].copy().sort_values('month').set_index('month')
    y = df_prod['quantity']
    
    try:
        model = sm.tsa.statespace.SARIMAX(
            y,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        pred = results.get_prediction(start=y.index[0], end=y.index[-1]).predicted_mean
        
        tmp = pd.DataFrame({
            'month': y.index,          
            'actual': y.values,
            'predicted': pred.values,
            'product_id': pid
        })
        actual_pred_list.append(tmp)

    except:
        pass



eval_df = pd.concat(actual_pred_list)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))


# Actual vs Predicted scatter plot

axes[0].scatter(eval_df['actual'], eval_df['predicted'], alpha=0.3, s=10, color='steelblue')
lim = max(eval_df['actual'].max(), eval_df['predicted'].max())
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1)
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('SARIMAX — Actual vs Predicted')


# MAPE Distribution

axes[1].hist(sarimax_eval['mape'].dropna(), bins=20, color='steelblue', edgecolor='white')
mean_mape = sarimax_eval['mape'].mean()
axes[1].axvline(mean_mape, color='red', linestyle='--', label=f'Mean MAPE {mean_mape:.2f}')
axes[1].set_xlabel('MAPE')
axes[1].set_ylabel('Products')
axes[1].set_title('SARIMAX — MAPE Distribution')
axes[1].legend()


# MAE Over Time

monthly_mae = (
    eval_df
    .reset_index()
    .groupby('month')
    .apply(lambda g: np.mean(np.abs(g['actual'] - g['predicted'])))
    .reset_index(name='mae')
)

axes[2].plot(monthly_mae['month'], monthly_mae['mae'], 'o-', color='steelblue')
axes[2].set_xlabel('Month')
axes[2].set_ylabel('MAE')
axes[2].set_title('SARIMAX — MAE Over Time')
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle('SARIMAX — Forecast Accuracy', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Scatter plot loos reasonably aligned with diagonal, meaning the model is capturing general pattern but spread is wider than Prophet
# More points far from the diagonal
#variance increases at higher quantities
# SARIMAX is fitting the data, but not as tightly as Prophet.It’s likely struggling with:Zero‑heavy series
# Sparse products
# Irregular seasonality
# Products with short history
# This is normal — SARIMAX is sensitive to low‑volume, noisy series.
# Red flag in Mean MAPE: MAPE explodes when:Actual quantity = 0, Predicted quantity ≠ 0
# Since many products have months with zero demand, SARIMAX produces predictions on those months, and MAPE becomes meaningless.
# MAE fluctuates month to month: Some months spike sharply, No clear downward trend
# Error is inconsistent across the timeline
# SARIMAX is not stable over time.
# Conclusion: SARIMAX is not outperforming Prophet on your dataset.
# Prophet is more robust to:
# Sparse data
# Zero‑demand months
# Irregular seasonality
# Short histories
# As next step we can use other models like LightGBM or XGBoost, Replace MAPE with MAE, use auto-arima to tune parameters.