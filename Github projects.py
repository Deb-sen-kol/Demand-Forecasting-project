#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:14:12 2026

@author: debasmitasen
"""

import numpy as np
import pandas as pd

# We read te files and perform EDA to study the data and check for null values and outliers.
inv=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_inventoryNew.csv")
inv.shape
inv.describe()
inv.head(5)
inv.isnull().sum()

orditems=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_order_items.csv")
orditems.shape
orditems.describe()
orditems.head(5)
orditems.isnull().sum()

orders=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_orders.csv")
orders.shape
orders.describe()
pd.set_option('display.max_columns', None)
orders.head(5)
orders.isnull().sum()

products=pd.read_csv("/Users/debasmitasen/Downloads/blinkit_products.csv")
products.shape
products.describe()
products.head(5)
products.isnull().sum()

import matplotlib.pyplot as plt

inv.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
orditems.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
orders.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()
products.select_dtypes(include='number').boxplot(figsize=(12, 6))
plt.show()

# Prepare the data by joining the tables


df = orditems.merge(orders[['order_id','order_date','store_id']], on='order_id')
df = df.merge(products[['product_id','product_name','category']], on='product_id')

# 5,000 stores × 268 products = 1,340,000 possible combos but only 5,000 rows
print(df['order_date'].dt.to_period('M').nunique(), "months")  # should show ~19
print(df['store_id'].nunique(), "stores")
print(df['product_id'].nunique(), "products")
print(len(df), "total rows")

# check store order frequency

orders_per_store = df.groupby('store_id')['quantity'].count().sort_values(ascending=False)
print(orders_per_store.describe())
print(orders_per_store.head(10))  # busiest stores
# top 10 stores have 1 order each, so we forecast by product and not by store.

# Floor to day
df['order_date'].dtype
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M')
ts = df.groupby(['product_id', 'month'])['quantity'].sum().reset_index()

# Skip 1 product and keep 267 products
counts = ts.groupby(['product_id'])['quantity'].count()
print(f"Keeping  {(counts >= 6).sum()} products")
print(f"Skipping {(counts < 6).sum()} products")

ts['month'] = ts['month'].dt.to_timestamp()

# Only 1 product had 6+ months of data out of 268 products - data is very sparse at product level.

# How many months of data does each product have?
#counts = ts.groupby('product_id')['quantity'].count().sort_values(ascending=False)
#print(counts.head(20))
#print(counts.describe())
# min 5 months and max 17 months so most products have plenty data.So we lower the treshold to 5 months.
# even afyer lowering, only 1 product was forecasted
# so we troubleshoot
# Check what ts looks like
print(ts.shape)
print(ts.head())
print(ts['product_id'].nunique())
print(ts['month'].dtype)  # likely period[M]
print("Unique products in ts:", ts['product_id'].nunique())
print(ts.head())

#For time-series we use Prophet model
from prophet import Prophet
import pandas as pd
results = []
skipped = []

for product, group in ts.groupby('product_id'):
    print("Processing product:", product)

    temp = group[['month', 'quantity']].rename(columns={'month': 'ds', 'quantity': 'y'})
    temp['ds'] = pd.to_datetime(temp['ds'])

    # Check for NaN
    if temp['y'].isna().any():
        print("NaN in product", product)
        skipped.append(product)
        continue

    # Check for duplicates
    if temp['ds'].duplicated().any():
        print("Duplicate dates in product", product)
        skipped.append(product)
        continue

    # Check for too few months
    if len(temp) < 5:
        print("Too few months for", product)
        skipped.append(product)
        continue

    # 🔍 Check for all-zero series
    if temp['y'].sum() == 0:
        print("All-zero series:", product)
        skipped.append(product)
        continue

    # 🔍 Check for constant series
    if temp['y'].nunique() == 1:
        print("Constant series:", product)
        skipped.append(product)
        continue

    # Try fitting Prophet
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
        model.fit(temp)
    except Exception as e:
        print("Prophet failed for", product, "error:", e)
        skipped.append(product)
        continue

    # Forecast
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)
    forecast['product_id'] = product
    results.append(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'product_id']])

print("Number of product forecasts:", len(results))
predictions = pd.concat(results, ignore_index=True)
print("Unique products in predictions:", predictions['product_id'].nunique())
print(predictions['product_id'].value_counts().head())
