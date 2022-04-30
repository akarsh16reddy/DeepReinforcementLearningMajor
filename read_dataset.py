import pandas as pd

df = pd.read_excel('C:\Major_DRL\dataset_without_first_job.xlsx');

temp_df = df.loc[df['timestamp']==0]

temp_df1 = temp_df[['timestamp','rounded_load','type']]

temp_df2 = df.loc[:]

'''
for row in temp_df2.itertuples(index=True, name='Pandas'):
      print(row)
      '''
def get0TimeStepJobs():
    return temp_df2;