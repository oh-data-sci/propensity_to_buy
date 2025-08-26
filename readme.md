propensity to buy
===
# introduction
this is my attempt at a fun challenge from octopus ev. provided is a data set with a single table of leads, including whether they wound up making a purchase. the data set has 50k records and 14 columns, and only very few missing values. 

## the data file:
```
 0   lead_id                    int64         
 1   lead_entry_time            datetime64[us]
 2   has_placed_order           bool          
 3   time_order_placed          datetime64[us]
 4   salary                     float64       
 5   date_of_birth              datetime64[us]
 6   postcode                   object        
 7   n_engaged_minutes          float64       
 8   n_of_cars_viewed           int64         
 9   company_industry           object        
 10  day_of_week                int64         
 11  price_of_last_car_viewed   float64       
 12  step_reached_in_website    object        
 13  how_did_you_hear_about_us  object        
```

## structure of this repo
```
├── data
│   ├── company_industries.csv
│   ├── ev_sales_dataset.parquet              <--- source data, shared by octopus
│   └── propensity_to_buy.duckdb              <--- a duckdb database for access and analysis
├── notebooks
│   ├── 010_eda.ipynb                         <--- deep exploratory data analysis
│   ├── 020_feature engineering.ipynb         <--- converting raw data to relevant to task
│   ├── 030_modelling_baseline_lr.ipynb       <--- baseline model: logistic regression
│   ├── 031_modelling_rf.ipynb                <--- good old random forest. never fails. right?
|   └── 033_modelling_xb.ipynb                <--- xgboost + comparison of all three models
├── notes
│   ├── eda_report.html                       <--- eda  pre-feature engineering
│   ├── feature_engineered_eda_report.html    <--- eda post-feature engineering
│   └── Senior Data Scientist Task.pdf        <--- instructions
├── readme.md                                 <--- this file
├── requirements.txt                          <--- python libraries employed
├── sql
│   ├── create_db.sql                         <--- sql script to load the data
│   ├── eda.sql                               <--- early effort at eda, abandoned
│   └── feature_engineering.sql               <--- early attempt at feature engineering, abandoned
└── src                                       <--- helper functions called in the jupyter notebooks
    ├── __init__.py
    ├── eda.py
    ├── feature_engineering.py
    ├── pre_processing.py
    ├── train_logistic_regression.py        <--- model codes 
    ├── train_random_forest.py              <--- model codes 
    ├── train_xgboost.py                    <--- model codes 
    └── utils.py
```


# the challenge:
1. explore the data and learn something about the customers.
2. create a model predicting the propensity to buy.
3. compose a proposal to the head of sales. 

# the solution
i prepared a number of jupyter notebooks in the `notebooks/` folder. they are annotated code files, some with graphs and clarifying outputs. the justifications for the below is found in the notebooks.

1. data sets explored, and exploratory reports generated. insights learned along the way
    + about 1/8th of the leads made a (single) purchase.
    + leads can register negative salaries. a lead's salary is logically expected to be a factor in their propensity to purchase, but how negative incomes are to be interpreted ins uncertain. also, salaries and number of engaged minutes are two columns with missing values. (time when order placed has as many populated cells as there are purchases, so more values are not expected in that field).
    + average salary of purchasers significantly higher (£76k) than the overall population (£57k) and especially higher than the average salary of the unconverted population (£53k).
    + for the buyers, there is a marked difference between the low volume on monday (weekday=0) and higher than daily average counts on wednesday-thursday and sunday. the overall propensity to buy ranges from 11.5% for lead entry on a monday to 13.8% on a wednesday. 
    + outside of clearly erroneous session lengths, 2 hours 20 minutes is the maximum time spent. buyers genereally less likely to spend 10 minutes or less than unconverted. compared to non buyers, more buyers spend between 20 and 30 minutes on the site.
    + only 8 area codes account for the bulk of sales, but the rest of the sales are evenly spread amoung the remaining 117.
    + as could be guessed, buyers are more likely to view a number of cars (>10).
    + the leads are more or less evenly distributed amoung the industries in the data set. no obvious difference could be spotted between the industries of the buyers and the industries of the unvonverted.
    + leads from google and other outperform leads from a company benefits page.

2. three statistical models created, one linear (logistic regression) and two-tree based ensamble models. none of them managed to pass muster, at best getting ~82% recall (number of high odds leads surfaced) but only measly 37% in precision (ratio of surfaced leads actually making a purchase).
    + recall is the more important metric to target so as to not miss any potential buyer, but the precision must also be higher so that there are fewer uninterested leads being surfaced for follow up and prodding.
    + crossvalidation might help some, but it is likely that the data is not capturing the required drivers that 

3. a proposal to the head of sales:
    + we have been looking into leads data in order to build a model that can predict the odds of a lead becoming a buyer.
    + we are not at a stage where we can reliably predict high odds of conversion using this model, however, that is not to say that there is nothing to be learned from it.
    + analysing the data we have, we have spotted some features of leads that elevate their odds of conversion. 
    + chief among these features is salary, unsurprisingly. the higher the salary the higher the propensity to buy.
        + relatedly, we saw that the higher the "affordability ratio" (between a lead's salary and the price of the car they last viewed) to increase the odds of conversion.
    + we also saw that their engagement (i.e. duration of their browsing session and the stage they reached in it) influencing the odds, suggesting that flattening of the site hierarchy (so that it does not take many clicks to get deep in, and see many options) could help boost conversion.

# proposed next steps
in the limited time, this is limited progress. what i suggest to do next:

- re-implement the feature engineering in sql so that it can run where the data reside
- further data wrangling, towards building more enhanced features, based on interaction of present features
- cross validated models, to better explore the parameter space to find a better fit.
- neural network models, potentially more flexible
