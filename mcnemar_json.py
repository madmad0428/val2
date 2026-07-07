import pandas as pd
from scipy.stats import chi2

df = pd.read_csv('experiment_results_json_v3.csv', dtype=str)

def mcnemar(col1, col2, name1, name2):
    a = df[col1].values
    b = df[col2].values
    b_count = sum((a=='1') & (b=='0'))
    c_count = sum((a=='0') & (b=='1'))
    chi2_stat = (abs(b_count - c_count) - 1)**2 / (b_count + c_count)
    p = 1 - chi2.cdf(chi2_stat, df=1)
    sig = "유의미 p<0.05" if p<0.05 else "유의미하지않음"
    print(f'{name1} vs {name2}: chi2={chi2_stat:.3f}, p={p:.4f} [{sig}]')

mcnemar('Correct_CoT','Correct_FewShot','CoT','FewShot')
mcnemar('Correct_CoT','Correct_Role','CoT','Role')
mcnemar('Correct_Role','Correct_FewShot','Role','FewShot')