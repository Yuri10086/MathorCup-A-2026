from pathlib import Path
import json
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent/'results'
stats=json.loads((out/'q3_censor_repeated_summary.json').read_text())['by_seed'];df=pd.DataFrame(stats).sort_values('seed');i=np.arange(len(df));w=.36
fig,ax=plt.subplots(figsize=(8,4.6));ax.bar(i-w/2,df.baseline,w,label='Gaussian only');ax.bar(i+w/2,df.censor_aware,w,label='Floor-aware');ax.set_xticks(i,[str(x) for x in df.seed]);ax.set_ylabel('Held-out floor frequency MAE');ax.set_xlabel('Style-stratified split seed');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'figure_floor_calibration.png',dpi=160);plt.close(fig)
tr=pd.read_csv(out/'pils_to_tripel_optimized_audit_11.csv');x=np.arange(len(tr));fig,ax=plt.subplots(figsize=(9,4.7));ax.plot(x,tr.distance_to_target_before,'o-',label='Before');ax.plot(x,tr.distance_to_target_after,'o-',label='After constrained transfer');ax.set_xticks(x,tr.beer.astype(str),rotation=45);ax.set_xlabel('Source beer');ax.set_ylabel('Latent distance to Tripel center');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'figure_transfer_distance.png',dpi=160);plt.close(fig)
profile=pd.read_csv(out/'flavor_fingerprint_by_style.csv');cols=['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all','acidity','bitternes','sweetness','alcohol','body','overall'];p=profile.set_index('style')[cols]
fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(p.to_numpy(),aspect='auto',vmin=0,vmax=6);ax.set_yticks(np.arange(len(p)),p.index,fontsize=8);ax.set_xticks(np.arange(len(cols)),cols,rotation=60,ha='right',fontsize=8);fig.colorbar(im,ax=ax,label='Predicted sensory score 0–6',shrink=.85);fig.tight_layout();fig.savefig(out/'figure_style_fingerprint.png',dpi=160);plt.close(fig)
print('3 figures written')
