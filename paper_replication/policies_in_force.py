import numpy as np
import pandas as pd
import os

### *** HELPER FUNCTIONS *** ###

def policies_in_force_over_time(df,count_col='policyCount'):
    
    """
    This function takes as input a policy-level dataframe (1 row = 1 policy) 
    and converts it into a timeseries of the number of policies in force on each
    day based on the start/end dates of each policy. 

    param: df: policy-level dataframe 
    param: count_col: column representing the "weight" that a given record contributes to the count.
                      If you want to count all records, then this should correspond to a column of ones. 
                      If you want to count only specific recors (e.g., SFHA policies) then this can contain 
                      a combination of zeros and ones. 
    """

    start_date = '2009-01-01'
    end_date = pd.Timestamp('today').strftime('%Y-%m-%d')

    t = pd.date_range(start_date,end_date,freq='D').astype('date32[day][pyarrow]')
    timeseries_df = pd.DataFrame(data={'date':t})

    inflow = df[['policyEffectiveDate',count_col]].groupby('policyEffectiveDate').sum().reset_index().rename(columns={count_col:'inflow','policyEffectiveDate':'date'})
    outflow = df[['policyTerminationDate',count_col]].groupby('policyTerminationDate').sum().reset_index().rename(columns={count_col:'outflow','policyTerminationDate':'date'})

    timeseries_df = pd.merge(timeseries_df,inflow,on='date',how='left')
    timeseries_df = pd.merge(timeseries_df,outflow,on='date',how='left')
    timeseries_df.fillna(0,inplace=True)

    timeseries_df['netflow'] = timeseries_df['inflow'] - timeseries_df['outflow']
    timeseries_df['policies_in_force'] = timeseries_df['netflow'].cumsum()

    return(timeseries_df)

### *** INITIAL SETUP *** ###

# Create folder for output
pwd = os.getcwd()
outfolder = os.path.join(pwd,'PIF_by_state')
if not os.path.exists(outfolder):
    os.makedirs(outfolder,exist_ok=True)

# Determine which state to process
state_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
state_list = np.loadtxt('US_state_list.txt',dtype=str)
state = state_list[state_idx]
print(state,flush=True)

### *** LOAD DATA *** ###

## OpenFEMA NFIP policies v2
# Available at: https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2
usecols = ['policyEffectiveDate',
           'policyTerminationDate',
           'originalNBDate',
           'propertyState',
           'countyCode',
           'reportedZipCode',
           'ratedFloodZone',
           'occupancyType']

date_cols = ['policyEffectiveDate','policyTerminationDate','originalNBDate']

filters = [('propertyState','=',state)]

policies = pd.read_parquet('FimaNfipPolicies.parquet',columns=usecols,filters=filters).rename(columns={'reportedZipCode':'zipCode'})
policies[date_cols] = policies[date_cols].astype('date32[day][pyarrow]')

### *** IMPLEMENT DATA QUALITY CONTROL CHECKS *** ###

original_n = len(policies)
policies = policies.dropna()
policies = policies[(policies['zipCode'].str.isdigit())]
policies['zipCode'] = policies['zipCode'].apply(lambda x: f'{int(x):05d}')
final_n = len(policies)

print(f'{final_n} / {original_n} ({100*final_n/original_n:.2f}%) of policy records passed quality checks.')

### *** CALCULATE NUMBER OF POLICIES IN FORCE OVER TIME *** ###

# Filter out non-residential policies
non_residential_occupancy_types = [4,6,17,18,19]
policies = policies[~policies['occupancyType'].isin(non_residential_occupancy_types)]

# Create indicator denoting whether record is associated with new or existing policyholder
policies['newPolicyIndicator'] = (policies['policyEffectiveDate'] == policies['originalNBDate']).astype(int)

# Create special flood hazard area (SFHA) indicator
# (encompasses flood zones starting with A or V) 
policies['sfhaIndicator'] = (policies['ratedFloodZone'].str.startswith('A')|policies['ratedFloodZone'].str.startswith('V')).astype(int)

# Create indicators for combo of new/existing and SFHA/non-SFHA

policies['newPIF'] = policies['newPolicyIndicator']
policies['newSfhaPIF'] = policies['newPolicyIndicator']*policies['sfhaIndicator']
policies['newNonSfhaPIF'] = policies['newPolicyIndicator']*(1-policies['sfhaIndicator'])

policies['existPIF'] = (1-policies['newPolicyIndicator'])
policies['existSfhaPIF'] = (1-policies['newPolicyIndicator'])*policies['sfhaIndicator']
policies['existNonSfhaPIF'] = (1-policies['newPolicyIndicator'])*(1-policies['sfhaIndicator'])

# Create groupby object that we can use to stratify our policy counts
strat_cols = ['propertyState','countyCode','zipCode']
G = policies.groupby(strat_cols)

## Calculate number of policies-in-force within each strata

# New policies
outcome_var = 'newPIF'
newPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
newPIF = newPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# New SFHA policies
outcome_var = 'newSfhaPIF'
newSfhaPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
newSfhaPIF = newSfhaPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# New Non-SFHA
outcome_var = 'newNonSfhaPIF'
newNonSfhaPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
newNonSfhaPIF = newNonSfhaPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# Existing policies
outcome_var = 'existPIF'
existPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
existPIF = existPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# Existing SFHA policies
outcome_var = 'existSfhaPIF'
existSfhaPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
existSfhaPIF = existSfhaPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# Existing Non-SFHA policies
outcome_var = 'existNonSfhaPIF'
existNonSfhaPIF = G.apply(policies_in_force_over_time,count_col=outcome_var)
existNonSfhaPIF = existNonSfhaPIF.reset_index()[strat_cols + ['date','policies_in_force']].rename(columns={'policies_in_force':outcome_var})

# Merge datasets
combined_df = pd.merge(newPIF,newSfhaPIF,on=strat_cols + ['date'],how='outer')
combined_df = pd.merge(combined_df,newNonSfhaPIF,on=strat_cols + ['date'],how='outer')
combined_df = pd.merge(combined_df,existPIF,on=strat_cols + ['date'],how='outer')
combined_df = pd.merge(combined_df,existSfhaPIF,on=strat_cols + ['date'],how='outer')
combined_df = pd.merge(combined_df,existNonSfhaPIF,on=strat_cols + ['date'],how='outer')

# To save on space, only record number of policies in force at end of month
combined_df['period'] = pd.to_datetime(combined_df['date']).dt.to_period('M')
combined_df['period_end'] = combined_df['period'].dt.end_time
combined_df['period_end'] = combined_df['period_end'].astype('date32[day][pyarrow]')
combined_df = combined_df[combined_df['date'] == combined_df['period_end']].drop(columns=['period','period_end'])

# Subsets to dates where OpenFEMA captures full policy base in force
mask = (combined_df['date'] >= pd.Timestamp('2010-01-01'))&(combined_df['date'] < pd.Timestamp('2025-10-01'))
combined_df = combined_df[mask].reset_index(drop=True)

### *** SAVE RESULTS *** ###

combined_df.rename(columns={'propertyState':'state'},inplace=True)

data_types = {'state':'string[pyarrow]',
              'countyCode':'string[pyarrow]',
              'zipCode':'string[pyarrow]',
              'date':'date32[day][pyarrow]',
              'newPIF':'int64[pyarrow]',
              'newSfhaPIF':'int64[pyarrow]',
              'newNonSfhaPIF':'int64[pyarrow]',
              'existPIF':'int64[pyarrow]',
              'existSfhaPIF':'int64[pyarrow]',
              'existNonSfhaPIF':'int64[pyarrow]'}

combined_df = combined_df.astype(data_types)

outname = os.path.join(outfolder,f'{state}_policies_in_force.parquet')
combined_df.to_parquet(outname)