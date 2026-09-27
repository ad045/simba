import pandas as pd

path = "output/gnm/hcp_schaefer_100_dataset/11_mst_2500_animal_0/summary_indiv_energy_for_exp_05_mst_animal_0_compared_with_hcp_schaefer_100.csv"

df = pd.read_csv(path)
energy_cols = [c for c in df.columns if c.startswith('MaxCrit_subject_')]
df['mean_energy'] = df[energy_cols].mean(axis=1)
best = df.loc[df['mean_energy'].idxmin()]
print(best.to_string())
