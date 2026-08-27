import sys, numpy as np, pandas as pd, torch
from pathlib import Path
ROOT=Path("/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_benchmarking"); sys.path.insert(0,str(ROOT))
from experiments_config import MORPHO_DIR, MORPHO_EXP, SELECTED_MEASURES, CONSENSUS_PATH, DIST_MATRIX_PATH
from experiment_structural_gradient.run_structural_gradient import load_net
from src.comparing_connectomes.frobenius_comparer import FrobeniusEvaluator
from src.comparing_connectomes.energy_comparer import EnergyEvaluator
from src.comparing_connectomes.portrait_divergence_comparer import PortraitDivergence

SP=Path("/private/tmp/claude-501/-Users-adrian-Desktop-Benchmarking-/9f056563-0b5a-4960-afe7-7a67f9d2a132/scratchpad")
T = {"paper_495": np.load(CONSENSUS_PATH)[0].astype(float),
     "real_594":  np.load(SP/"consensus_12pct.npy")}
names=list(T); tgt=torch.tensor(np.stack([T[n] for n in names]),dtype=torch.float32)

pool=pd.read_csv(MORPHO_DIR/f"summary_indiv_{SELECTED_MEASURES[0]}_for_exp_{MORPHO_EXP}.csv")[["filename","eta","gamma","id"]]
pool=pool[pool["id"]<2]
etas=np.sort(pool.eta.unique())[::3]; gammas=np.sort(pool.gamma.unique())[::3]
pool=pool[pool.eta.isin(etas)&pool.gamma.isin(gammas)]
print("cells",len(etas)*len(gammas),"networks",len(pool))

D=np.load(DIST_MATRIX_PATH)
from gnm import evaluation as gnm_eval
dist_t=torch.tensor(D,dtype=torch.float32)
energy_criteria=gnm_eval.MaxCriteria([gnm_eval.DegreeKS(),gnm_eval.ClusteringKS(),
                                      gnm_eval.EdgeLengthKS(dist_t),gnm_eval.BetweennessKS()])
evals={"frobenius":FrobeniusEvaluator(),"energy":EnergyEvaluator([energy_criteria]),
       "portrait":PortraitDivergence()}
rows=[]
for _,r in pool.iterrows():
    g=torch.tensor(load_net(r.filename),dtype=torch.float32).unsqueeze(0)
    rec={"eta":r.eta,"gamma":r.gamma}
    for m,ev in evals.items():
        out=ev(g,tgt)
        for i,n in enumerate(names): rec[f"{m}__{n}"]=float(out[i])
    rows.append(rec)
df=pd.DataFrame(rows); df.to_csv(SP/"density_check_real12.csv",index=False)
land=df.groupby(["eta","gamma"]).mean().reset_index()
from scipy.stats import spearmanr
for m in evals:
    print("\n==",m)
    for n in names:
        c=land.loc[land[f"{m}__{n}"].idxmin()]
        print("  target %-9s best fit eta=%6.2f gamma=%5.2f"%(n,c.eta,c.gamma))
    for a,b in [("paper_495","real_594")]:
        rho=spearmanr(land[f"{m}__{a}"],land[f"{m}__{b}"]).statistic
        print("  spearman %s vs %s: %.4f"%(a,b,rho))
