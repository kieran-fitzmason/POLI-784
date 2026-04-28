import numpy as np
import pandas as pd
import os

### *** LOAD DATA SOURCES *** ###

# Get current directory
pwd = os.getcwd()

## NFIP policies-in-force by zip code
policies_dir = os.path.join(pwd,'PIF_by_state')
policies_path_list = [os.path.join(policies_dir,x) for x in np.sort(os.listdir(policies_dir))]
policies = pd.read_parquet(list(policies_path_list))

## Change in median premiums by zip code under Risk Rating 2.0
# Available at: https://www.fema.gov/flood-insurance/work-with-nfip/risk-rating/single-family-home
data_types = {'State':str,
              'Zip Code':str,
              'Median Current Cost of Insurance':float,
              'Median Risk-based Cost of Insurance':float}

rename_dict = {'State':'state',
               'Zip Code':'zipCode',
               'Median Current Cost of Insurance':'medianLegacyPremium',
               'Median Risk-based Cost of Insurance':'medianFullRiskPremium'}

premium_increases = pd.read_csv('NFIP_RR2_premium_increases.csv',usecols=data_types.keys(),dtype=data_types)
premium_increases.rename(columns=rename_dict,inplace=True)

## Median household income by ZCTA
# Available at: https://data.census.gov/table?q=DP03&g=010XX00US$8600000&y=2022
data_types = {'GEO_ID':str,
              'DP03_0062E':str}

rename_dict = {'GEO_ID':'zipCode',
               'DP03_0062E':'medianHouseholdIncome'}

income = pd.read_csv('ACSDP5Y2022.DP03-Data.csv',usecols=data_types.keys(),dtype=data_types)
income.rename(columns=rename_dict,inplace=True)

# Drop description header row
income.drop(0,inplace=True)

### *** DATA CLEANING *** ###

## Premium increases

original_n = len(premium_increases)
premium_increases = premium_increases.dropna()
premium_increases = premium_increases[(premium_increases['zipCode'].str.isdigit())]
premium_increases['zipCode'] = premium_increases['zipCode'].apply(lambda x: f'{int(x):05d}')
final_n = len(premium_increases)

print(f'{final_n} / {original_n} ({100*final_n/original_n:.2f}%) of premium increase records passed quality checks.')

## Income
original_n = len(income)
income['zipCode'] = income['zipCode'].apply(lambda x: x.split('860Z200US')[1])
income['medianHouseholdIncome'] = pd.to_numeric(income['medianHouseholdIncome'],errors='coerce')
income = income.dropna()
final_n = len(income)

print(f'{final_n} / {original_n} ({100*final_n/original_n:.2f}%) of household income records passed quality checks.')

### *** LINK POLICIES IN FORCE WITH PREMIUM INCREASES *** ###

# Drop zip codes for which we do not have data on premium increases
policies = policies[policies['zipCode'].isin(premium_increases['zipCode'].unique())]

# Merge PIF data with premium increase data
policies = pd.merge(policies,premium_increases,on=['state','zipCode'],how='left').dropna()

# Calculate relative change in premium
policies['relativeIncreasePremium'] = policies['medianFullRiskPremium']/policies['medianLegacyPremium'] - 1

# Discretize treatment level using same bins as Gourevitch et al. (2025)
treatment_bins = [0.08,0.34,0.94,np.inf]
policies['treatmentLevel'] = np.digitize(policies['relativeIncreasePremium'],treatment_bins)

### *** LINK POLICIES IN FORCE WITH INCOME DATA *** ###

# Discretize income based on 25th, 50th, and 75th percentile of ZCTA median income
income_bins = income['medianHouseholdIncome'].quantile([0.25,0.5,0.75]).tolist() + [np.inf]
income['incomeLevel'] = np.digitize(income['medianHouseholdIncome'],income_bins)

# Attach data to policies
policies = pd.merge(policies,income,on='zipCode',how='left')

### *** CALCULATE TIME RELATIVE TO RR2.0 ROLLOUT *** ###

policies['period'] = pd.to_datetime(policies['date']).dt.to_period('M')

# Phase I of rollout: October 2021
phaseI_period = pd.Period('2021-10')

# Phase II of rollout: April 2022
phaseII_period = pd.Period('2022-04')

# Calculate time offset
policies['monthsRelativeToOct2021'] = (policies['period'] - phaseI_period).apply(lambda x: x.n)
policies['monthsRelativeToApr2022'] = (policies['period'] - phaseII_period).apply(lambda x: x.n)

# Filter to 12 months before and 36 months after RR2.0 rollout
policies = policies[policies['monthsRelativeToOct2021'] >= -12]
policies = policies[policies['monthsRelativeToApr2022'] <= 36]

### *** SAVE FINAL DATASET *** ###

# Set data types for more efficient storage
data_types = {'state':'string[pyarrow]',
              'assignedCountyCode':'string[pyarrow]',
              'zipCode':'string[pyarrow]',
              'date':'date32[day][pyarrow]',
              'monthsRelativeToOct2021':'int32[pyarrow]',
              'monthsRelativeToApr2022':'int32[pyarrow]',
              'newPIF':'int32[pyarrow]',
              'newSfhaPIF':'int32[pyarrow]',
              'newNonSfhaPIF':'int32[pyarrow]',
              'existPIF':'int32[pyarrow]',
              'existSfhaPIF':'int32[pyarrow]',
              'existNonSfhaPIF':'int32[pyarrow]',
              'medianLegacyPremium':'float32[pyarrow]',
              'medianFullRiskPremium':'float32[pyarrow]',
              'relativeIncreasePremium':'float32[pyarrow]',
              'treatmentLevel':'int32[pyarrow]',
              'medianHouseholdIncome':'float32[pyarrow]',
              'incomeLevel':'int32[pyarrow]'}

policies = policies[data_types.keys()].astype(data_types).reset_index(drop=True)

# Save as parquet file
policies.to_parquet('NFIP_RR2_panel_data.parquet')