import duckdb
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

with duckdb.connect('data/propensity.duckdb') as con:
    sales_df = con.sql("SELECT * FROM sales;").df()

print(sales_df.info())
profile = ProfileReport(sales_df, title="profiling report -- oskar holm")
profile.to_file("../notes/eda_report.html")