# # ZIMove:
# ## Anomalies analysis & manipulations for logistic cycles table
print('start of script')
### requirements
import logging
from datetime import datetime
import os
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
from nltk import ngrams
###
print('done imports')
print(datetime.now())

#Define logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
chandler = logging.StreamHandler()
chandler.setFormatter(formatter)
logger.addHandler(chandler)
logger.info('Start reading data')
###

os.chdir("C:/Users/retzer.matan/Desktop/Logistics")
data=pd.read_csv("ZZIM_MOVE_CYC_WIC_TEST.csv")

logger.info('Done reading data')

# ### Data manipulation

data['Disch. Dat']=pd.to_datetime(data['Disch. Dat'], format='%d-%m-%y')
data['Cntry. End Date']=pd.to_datetime(data['Cntry. End Date'], format='%d-%m-%y')
data['Cntry. Start Date']=pd.to_datetime(data['Cntry. Start Date'], format='%d-%m-%y')
data['Cycle Last Move Date']=pd.to_datetime(data['Cycle Last Move Date'], format='%d-%m-%y')
#
data['Cycle Movements'].replace(" ", "", inplace=True, regex=True)

data.drop_duplicates(keep=False,inplace=True) #remove duplications

logger.info('Done data manipulations')

df=data.drop('Cycle Countries Counter',axis='columns', inplace=False)

df.rename({'Cyc._Last_Mov' : 'Cyc_Last_Mov'}, axis=1, inplace=True)

# ### Adding new fields

df['Cycle_Period']=df['Cycle Last Move Date']-df['Disch. Dat']
df['Cycle_Period']= (df['Cycle_Period']).dt.days
df['last_move_cat']=np.where(df['Cycle Movements'].str.contains('NONMOVE'),"OPEN","CLOSE")
df['Period Move']=df['Cycle_Period']/df['Move Count']
df['Unused_Container']=np.where((((df['Cycle Days']-df['Cycle_Period'])>=180)&
(df['last_move_cat']=="OPEN")),1,0)
df['Days_After_Dmge']=np.where(((df['Cyc. Last Mov']=="DMGE")&
(df['last_move_cat']=="OPEN")),(df['Cycle Days']-df['Cycle_Period']),0)
#
df['Cycle_Period']=df['Cycle_Period'].round(0)

logger.info('Done new fields data')
# ### Adding anomaly detection features

STDS = 2  # Number of standard deviation that defines 'outlier'

z = df[['Country', 'last_move_cat', 'Cycle Days', 'Period Move', 'Move Count',
 'Days_After_Dmge']].groupby(['Country', 
 'last_move_cat']).transform(lambda group: (group - group.mean()).div(group.std()))

outliers = z.abs() > STDS

# rename columns
outliers.rename({'Cycle Days': 'Cycle_Days_Anomaly', 'Days_After_Dmge': 'DMGE_Late_Anomaly',
'Move Count': 'Move_Count_Anomaly', 'Period Move': 'Period_Move_Anomaly'}, axis=1,inplace=True)

logger.info('Done outliers features')
# concat df with outliers columns
df_new=pd.concat([df.reset_index(drop=True), outliers], axis=1)

# #### Cycle period anomaly features

round(float(np.float64(df['Cycle_Period'].mean())), 0)

### mead and std functions
def r_mean(i):
    return round(float(np.float64(i.mean())), 0)

def r_std(i):
    return round(float(np.float64(i.std())), 0)

#
df_new_period=pd.DataFrame(df_new.groupby(['Country', 'last_move_cat','Disch. Mov','Move Count'])
['Cycle_Period'].agg([r_mean,r_std]).reset_index())

df_new_period['mean_0.1']=round(df_new_period.r_mean*0.1,0)

df_new_period['diff_below']=np.where((df_new_period['r_mean']-(2*df_new_period['r_std']))>1,
(df_new_period['r_mean']-(2*df_new_period['r_std'])),df_new_period['mean_0.1'])
df_new_period['diff_above']=(df_new_period['r_mean']+(2.5*df_new_period['r_std']))

df_new_period=df_new_period.dropna() #remove na

###merge datasets
df_new=pd.merge(df_new, df_new_period,  how='left', on=['Country', 'last_move_cat','Disch. Mov','Move Count'])

###update feature
df_new['Period_Anomaly']=np.where(df_new['Cycle_Period']<df_new['diff_below'],'Short',
(np.where(df_new['Cycle_Period']>df_new['diff_above'],'Long','')))
###drop columns
df_new=df_new.drop(['r_mean','r_std','mean_0.1','diff_below','diff_above'],axis='columns', inplace=False)

logger.info('Done cycle period features')

# ### NLP Features

# #### Add anomaly movements

# create anomaly moves table
moves_anomaly = df_new.assign(words=df_new['Cycle Movements'].str.upper().str.split(',')).explode('words').groupby('Country')['words'].value_counts().reset_index(name='counts')
#
moves_freq=pd.DataFrame(moves_anomaly[moves_anomaly['words']!='NONMOVE'])

#create freq column
moves_freq['move_pct'] = moves_freq['counts'] / moves_freq.groupby('Country')['counts'].transform('sum')

moves_freq=pd.DataFrame(moves_freq[moves_freq['move_pct']<=0.0001])

logger.info('Done anomaly movements')

# #### Create new function to build unfrequent moves field

countries = set(df_new.Country.unique())#create group by country

###
country_unfreq_moves_dict = defaultdict(set)
for c in countries:
    c_df = moves_freq.loc[moves_freq['Country'] == c]
    country_unfreq_moves_dict[c] = set(c_df.words.unique())


def get_unfrequent_moves(row, country_unfreq_moves_dict):
    found = []
    for move in row['Cycle Movements'].split(','):
        if move in country_unfreq_moves_dict[row['Country']]: found.append(move)
    return ','.join(found)

###create the new column
df_new['Special_move'] = df_new.apply(lambda row : get_unfrequent_moves(row, country_unfreq_moves_dict), axis = 1)

logger.info('Done unfrequent moves')

# #### Add count of anomaly moves field

df_new['Special_move_count']= np.where(df_new['Special_move'].str.split().str.len()==0,0,
(np.where(df_new.Special_move.str.count(',')==0,1,df_new.Special_move.str.count(',')+1)))

# #### Add anomaly cycles (sequences)

#create cycle freq table
cycle_anomaly = df_new.groupby('Country')['Cycle Movements'].apply(lambda x: x.value_counts()).reset_index(name='counts')

#rename the field
cycle_anomaly.rename({'level_1': 'Cycles'}, axis=1,inplace=True)

###create freq field
cycle_anomaly['cycle_pct'] = cycle_anomaly['counts'] / cycle_anomaly.groupby('Country')['counts'].transform('sum')
#
cycle_freq=pd.DataFrame(cycle_anomaly[cycle_anomaly['cycle_pct']<=0.0001])
#
cycle_anomaly=cycle_freq.drop('cycle_pct',axis='columns', inplace=False)

#merge the df with least freq cycles table
df_new = pd.merge(df_new, cycle_anomaly,  how='left', left_on=['Country',
'Cycle Movements'], right_on = ['Country','Cycles'])

#create new column
df_new['Cycle_anomaly']=np.where(df_new['counts'].isna(),0,1)

#drop unneccecery fileds
df_new=df_new.drop(['Cycles','counts'],axis='columns', inplace=False)

logger.info('Done anomaly moves count')

# #### Add anomaly move pairs (sequence)

def get_move_ngram_list(doc, k):
    broken_doc = [move for move in doc.split(',') if move != 'NONMOVE']
    move_ngrams = ['_'.join(t) for t in ngrams(broken_doc,k)]
    return move_ngrams

def get_ngrams_counter(docs, k):
    cnt = Counter()
    for doc in docs:
        cnt.update(get_move_ngram_list(doc, k))
    return cnt

#

ngrams_list = [2, 3, 4, 5]
country_ngrams_move_dict = defaultdict(dict)
for c in countries:
    #print('Country: ', c)
    country_move_docs = df_new.loc[df_new['Country'] == c]['Cycle Movements'].values
    for n in ngrams_list:
        country_ngrams_move_dict[c][n] = get_ngrams_counter(country_move_docs, n)

#
logger.info('Done move pairs')

country_ngram_total_dict = defaultdict(dict)
for c in countries:
    #print('Country: ', c)
    for n in ngrams_list:
        n_total = sum(country_ngrams_move_dict[c][n].values())
        country_ngram_total_dict[c][n] = n_total
#         print(f'total {n}-grams: ', n_total)
#country_ngram_total_dict

# creates a dictionary in the following template:
# Country code -> n-gram [2,5] -> frequency dicit (k = ngram, v = frequency)
country_ngrams_move_freq_dict = defaultdict(dict)
for c in countries:
    for n in country_ngrams_move_dict[c]:
        freq_dict = {}
        for k,v in country_ngrams_move_dict[c][n].items():
            freq_dict[k] = v/country_ngram_total_dict[c][n]
        country_ngrams_move_freq_dict[c][n] = freq_dict

logger.info('Done n-grams functions')

# returns the least frequent ngrams
FREQ_THRESHOLD = 0.00001
country_ngrams_anomly_seq_move_dict = defaultdict(dict)
for c in countries:
    #print(f'{c}:')
    for n, freq_dict in country_ngrams_move_freq_dict[c].items():
        #print(f'{n}-grams:\n')
        least_freq = {}
        for k,v in freq_dict.items():
            if v <= FREQ_THRESHOLD: least_freq[k] = v
        country_ngrams_anomly_seq_move_dict[c][n] = least_freq
        #print(f'Least frequent {n}-grams: (total: {len(least_freq)}\n ', least_freq, '\n\n')
#

def get_anomly_seq_moves(row, n, country_ngrams_anomly_seq_move_dict):
    doc = row['Cycle Movements']
    anomly_ngrams = list(country_ngrams_anomly_seq_move_dict[row['Country']][n].keys())
    row_ngrams = get_move_ngram_list(row['Cycle Movements'], n)
    found = []
    for gram in row_ngrams:
        if gram in anomly_ngrams:
            found.append(gram)
    if found:
        return ','.join(found), len(found)
    else:
        return None, 0

logger.info('Done least freq n-grams')

# #### Create 2 fields for every n-gram (show & count)
### with the full capabilities: for n in [2,3,4,5]:

for n in [2,3]:
    df_new[f'Special_seq_{n}_moves'], df_new[f'Special_seq_{n}_count']  = zip(*df_new.apply(lambda row : 
    get_anomly_seq_moves(row, n, country_ngrams_anomly_seq_move_dict), axis = 1))

logger.info('done n-gram (show & count)')
# #### New feature: dmge,back count (>1)

df_new['Count_dmge_back_pairs'] = df_new['Cycle Movements'].str.count('DMGE,BACK')

# ### Anomaly score calc

df_new['Anomaly']=np.where(((df_new.Cycle_Days_Anomaly is False)&
(df_new.Period_Move_Anomaly is False)&(df_new.Move_Count_Anomaly is False)&
(df_new.DMGE_Late_Anomaly is False)&(df_new.Period_Anomaly=='')&
(df_new.Special_move_count==0)&(df_new.Cycle_anomaly==0)&
(df_new.Special_seq_2_count==0)&(df_new.Count_dmge_back_pairs<2)),0,1)

logger.info('End of script')

###export csv for testing:

df_new.to_csv("df_new_testing.csv", index=False)
print('End exporting data')
print(datetime.now())
######=========================================================== End of Script ====================================================#######