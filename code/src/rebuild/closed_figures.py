from q1 import *
from q2 import *
from q3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
COL={'Spearman':'#9BAABB','Hybrid2plus1':'#3775BA','Forward3':'#42949E','Redundancy':'#9A4D8E','TrainSelected':'#B64342','PCA_Gaussian':'#9BAABB','CopulaIndependent':'#3775BA','CopulaJoint':'#42949E','CVAE':'#B64342','Unconstrained':'#9BAABB','PointConstraint':'#3775BA','UncertaintyConstraint':'#42949E','Reject':'#B64342'}
SHORT={'PCA_Gaussian':'PCA Gaussian','CopulaIndependent':'Independent copula','CopulaJoint':'Joint copula','CVAE':'CVAE','Hybrid2plus1':'2+1','TrainSelected':'Train-selected','PointConstraint':'Point constraint','UncertaintyConstraint':'Uncertainty constraint','Unconstrained':'Unconstrained','Reject':'Reject'}
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(F/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def matrix(ax,mat,title,cbarlabel,fmtmax=None):
 cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#D5D5D5');im=ax.imshow(np.ma.masked_invalid(mat.to_numpy()),aspect='auto',cmap=cmap,vmin=0,vmax=fmtmax);ax.set_xticks(range(len(mat.columns)),[SHORT.get(v,v) for v in mat.columns],rotation=30,ha='right',fontsize=8);ax.set_yticks(range(len(mat.index)),mat.index,fontsize=8);ax.set_title(title);plt.colorbar(im,ax=ax,shrink=.75,label=cbarlabel)
def main():
 metrics=pd.read_csv(F/'q2/summary_metrics.csv').set_index('method').reindex(METHODS+['TrainSelected']);stab=pd.read_csv(F/'q2/stability_summary.csv').set_index('method').reindex(metrics.index);fig,ax=plt.subplots(1,3,figsize=(12,3.8));xx=np.arange(5);ax[0].bar(xx,metrics.MAE_mean,yerr=metrics.MAE_std,color=[COL[m] for m in metrics.index],capsize=3);ax[0].set_xticks(xx,[SHORT.get(m,m) for m in metrics.index],rotation=30,ha='right',fontsize=8);ax[0].set_ylabel('Raw panel mean MAE');ax[0].set_title('Two repeated 3-fold predictions');ax[1].bar(xx-.18,stab.compound_jaccard,width=.36,color='#3775BA',label='Compound');ax[1].bar(xx+.18,stab.group_jaccard,width=.36,color='#42949E',label='Alternative group');ax[1].set_xticks(xx,[SHORT.get(m,m) for m in metrics.index],rotation=30,ha='right',fontsize=8);ax[1].set_ylabel('Bootstrap Jaccard');ax[1].legend(fontsize=8);bc=pd.read_csv(F/'q2/bootstrap_choice_summary.csv');ax[2].barh([SHORT.get(m,m) for m in bc.method],bc['count'],color=[COL[m] for m in bc.method]);ax[2].set_xlabel('Choices among 100 bootstraps');ax[2].set_title('Selection uncertainty');save(fig,'10_q2_unified')
 df=pd.read_csv(F/'generation/metrics_22styles_summary.csv');models=C['q3_models'];fig,ax=plt.subplots(1,2,figsize=(11,7.5));m=df.pivot(index='style',columns='model',values='SWD_mean').reindex(index=ST,columns=models);matrix(ax[0],m,'Conditional distance (grey: no test beer)','SWD');m=df.pivot(index='style',columns='model',values='style_accuracy_mean').reindex(index=ST,columns=models);matrix(ax[1],m,'Independent classifier consistency','Fraction classified as target',1);save(fig,'11_generation_styles')
 fig,ax=plt.subplots(1,2,figsize=(11,7));m=df.pivot(index='style',columns='model',values='joint_floor_MAE_mean').reindex(index=ST,columns=models);matrix(ax[0],m,'Joint low-end indicator error','Pairwise probability MAE',1);m=df.pivot(index='style',columns='model',values='correlation_MAE_mean').reindex(index=ST,columns=models).dropna(how='all');matrix(ax[1],m,'Rank structure: common pairs; n_test >= 4','Spearman correlation MAE',1);save(fig,'12_generation_structure')
 m=pd.read_csv(F/'migration/per_beer.csv');policies=['Unconstrained','PointConstraint','UncertaintyConstraint','Reject'];fig,ax=plt.subplots(1,2,figsize=(11,4));ids=sorted(m.beer.unique());x=np.arange(9)
 for policy in policies:
  d=m[m.policy==policy].set_index('beer').loc[ids];ax[0].plot(x,d.target_gain,marker='o',ms=4,label=SHORT[policy],color=COL[policy]);ax[1].plot(x,d.stale_change,marker='o',ms=4,label=SHORT[policy],color=COL[policy])
 for a in ax:a.axhline(0,color='black',lw=.8);a.set_xticks(x,[str(i) for i in ids],fontsize=8);a.set_xlabel('Held-out source beer');
 ax[0].set_ylabel('Tripel classifier score gain');ax[1].set_ylabel('Independent stale-hops change');ax[0].legend(fontsize=8);save(fig,'13_migration_paired')
 fig,ax=plt.subplots(1,3,figsize=(11,3.7));summ=pd.read_csv(F/'migration/summary.csv');
 for a,metric,title in zip(ax,['independent_stale_increase','change_RMS','refused'],['Independent stale-hops increase rate','Scaled chemical RMS change','Rejection rate']):
  d=summ[summ.metric==metric].set_index('policy').loc[policies];a.bar(range(4),d['mean'],yerr=np.vstack([np.maximum(0,d['mean']-d.CI_low),np.maximum(0,d.CI_high-d['mean'])]),color=[COL[p] for p in policies],capsize=3);a.set_xticks(range(4),[SHORT[p] for p in policies],rotation=35,ha='right',fontsize=8);a.set_title(title,fontsize=10)
 save(fig,'14_migration_risk')
 # 按最低数字啤酒id修正的示例
 beer=ids[0];d=m[m.beer==beer];s=int(d.seed.iloc[0]);gen=joblib.load(F/f'models/generator_{pd.read_csv(O/"q3/selected.csv").set_index("seed").loc[s,"model"]}_{s}.joblib');i=int(np.flatnonzero(xdf.beer.to_numpy()==beer)[0]);source=gen.pre.imp.transform(X[[i]])[0];spec=pd.read_csv(F/'migration/spectra.csv');z=spec[spec.beer==beer].set_index('policy').loc[policies,chem].to_numpy();delta=(z-source)/gen.pre.scaler.scale_;known=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal'];ix=[chem.index(c) for c in known];other=[j for j in np.argsort(abs(delta).mean(0))[::-1] if j not in ix][:8];ix+=other;fig,ax=plt.subplots(figsize=(10,3.2));im=ax.imshow(delta[:,ix],aspect='auto',cmap='RdBu_r',vmin=-max(.1,abs(delta[:,ix]).max()),vmax=max(.1,abs(delta[:,ix]).max()));ax.set_yticks(range(4),[SHORT[p] for p in policies]);ax.set_xticks(range(len(ix)),[chem[j] for j in ix],rotation=35,ha='right',fontsize=8);ax.set_title(f'Fixed example beer {beer}: change in generator training SD units');fig.colorbar(im,ax=ax,label='Scaled change');save(fig,'15_transfer_chemistry');log('Closed-experiment publication figures complete')
if __name__=='__main__':main()
