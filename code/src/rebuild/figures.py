from q1 import *
from q2 import *
from q3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#3775BA';RED='#B64342';GREEN='#42949E'
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(O/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 q1=pd.read_csv(O/'q1/repeat_metrics.csv');q2=pd.read_csv(O/'q2/fold_metrics.csv');q3=pd.read_csv(O/'q3/test_metrics.csv');mg=pd.read_csv(O/'q3/migration_metrics.csv')
 fig,ax=plt.subplots(1,2,figsize=(10,4));st=pd.Series(style).value_counts();ax[0].barh(st.index[::-1],st.values[::-1],color=BLUE);ax[0].tick_params(axis='y',labelsize=7);ax[0].set_xlabel('Number of beers');floor=pd.read_csv(O/'audit/features.csv').nlargest(12,'minimum_fraction');ax[1].barh(floor.feature[::-1],floor.minimum_fraction[::-1],color=GREEN);ax[1].tick_params(axis='y',labelsize=7);ax[1].set_xlabel('Fraction at observed minimum');save(fig,'01_data')
 fig,ax=plt.subplots(1,2,figsize=(10,3.6));g=q1.groupby('model').MAE.agg(['mean','std']).sort_values('mean');ax[0].barh(g.index[::-1],g['mean'][::-1],xerr=g['std'][::-1],color=BLUE);ax[0].set_xlabel('Raw panel mean MAE (two repeats)');p=pd.read_csv(O/'q1/per_target.csv');v=p[p.model=='ET'].sort_values('MAE',ascending=False).head(12);ax[1].barh(v.target[::-1],v.MAE[::-1],color=GREEN);ax[1].set_xlabel('ET MAE by difficult sensory target');save(fig,'02_q1_accuracy')
 curves=pd.read_csv(O/'q1/training_curves.csv');fig,ax=plt.subplots(1,3,figsize=(10,3));
 for a,m in zip(ax,['MLP','Hierarchy','HierarchyRater']):
  z=curves[curves.model==m].groupby('epoch')[['train_mae','validation_mae']].mean();a.plot(z.index,z.train_mae,label='Training',color=BLUE);a.plot(z.index,z.validation_mae,label='Validation',color=RED);a.set_title(m);a.set_xlabel('Epoch');a.set_ylabel('MAE');a.legend(fontsize=8)
 save(fig,'03_q1_curves')
 fig,ax=plt.subplots(1,2,figsize=(10,3.6));sta=pd.read_csv(O/'q2/stability.csv');g=q2.groupby('model').MAE.mean().sort_values();ax[0].barh(g.index[::-1],g.values[::-1],color=BLUE);ax[0].set_xlabel('Mean fold MAE, 3 compounds');v=sta.groupby('method')[['compound_jaccard','group_jaccard']].mean();v.plot.bar(ax=ax[1],color=[BLUE,GREEN]);ax[1].tick_params(axis='x',rotation=25);ax[1].set_ylabel('Bootstrap Jaccard');ax[1].legend(['Compound','Alternative group'],fontsize=8);save(fig,'04_q2_selection')
 e=pd.read_csv(O/'q2/signed_edges.csv');G=nx.from_pandas_edgelist(e,'source','target',edge_attr='rho');pos=nx.spring_layout(G,seed=20261028,weight=None,k=.8,iterations=400);fig,ax=plt.subplots(figsize=(8,6));nx.draw_networkx_nodes(G,pos,node_size=[10+4*G.degree(n) for n in G],node_color=BLUE,ax=ax,alpha=.8);nx.draw_networkx_edges(G,pos,edge_color=[RED if G[u][v]['rho']<0 else '#A0B8CD' for u,v in G.edges],width=.7,alpha=.65,ax=ax);labels={n:str(i+1) for i,n in enumerate(sorted(G,key=G.degree,reverse=True)[:10])};ax.text(1.01,.9,'Highest-degree compounds (rank)\n'+'\n'.join(labels[n]+': '+n for n in labels),transform=ax.transAxes,fontsize=8,va='top');ax.set_title('Signed Spearman network, |rho| >= 0.65 (non-isolated nodes)');ax.axis('off');save(fig,'05_q2_network')
 fig,ax=plt.subplots(1,3,figsize=(11,3.5));
 for a,col,title in zip(ax,['SWD','correlation_MAE','floor_fraction_MAE'],['Distribution distance','Rank-dependence error','Minimum-mass error']):
  v=q3.groupby('model')[col].agg(['mean','std']);a.bar(np.arange(len(v)),v['mean'],yerr=v['std'],color=[BLUE,GREEN,'#8BCF8B',RED],capsize=3);a.set_xticks(range(len(v)),v.index,rotation=35,ha='right',fontsize=8);a.set_title(title)
 save(fig,'06_q3_quality')
 fig,ax=plt.subplots(1,2,figsize=(10,3.5));g=mg.assign(prob_change=mg.target_probability_after-mg.target_probability_before).groupby('method');a=g.prob_change.mean();b=g.independent_stale_change.mean();ax[0].bar(a.index,a.values,color=BLUE);ax[0].set_ylabel('Independent Tripel probability change');ax[1].bar(b.index,b.values,color=RED);ax[1].axhline(0,color='black',lw=.8);ax[1].set_ylabel('Independent stale-hops prediction change');
 for a in ax:a.tick_params(axis='x',rotation=20,labelsize=8)
 save(fig,'07_migration')
 sens=pd.read_csv(O/'q3/generated_sensory50.csv');cols=[c for c in ['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all','alcohol','body','sweet','bitter','stale_hops'] if c in sens];sel=['Tripel','Kriek','Lambic'];mat=sens.groupby('style')[cols].mean().loc[sel];fig,ax=plt.subplots(figsize=(9,2.7));im=ax.imshow(mat,vmin=0,vmax=6,cmap='YlGnBu',aspect='auto');ax.set_xticks(range(len(cols)),cols,rotation=35,ha='right');ax.set_yticks(range(3),sel);fig.colorbar(im,ax=ax,label='Predicted raw panel mean');save(fig,'08_generated_portraits')
 fig,ax=plt.subplots(1,2,figsize=(10,3.5));sen=pd.read_csv(O/'q1/sensitivity.csv');v=sen[sen.experiment.str.startswith('ET_mask')].groupby('experiment').MAE.mean();ax[0].plot([.05,.1,.2],[v[f'ET_mask_{p}'] for p in [.05,.1,.2]],marker='o',color=BLUE);ax[0].set_xlabel('Additional random missing fraction');ax[0].set_ylabel('ET MAE');cv=pd.read_csv(O/'q3/training_curves.csv').groupby('epoch')[['reconstruction','validation_reconstruction']].mean();ax[1].plot(cv.index,cv.reconstruction,label='Train',color=BLUE);ax[1].plot(cv.index,cv.validation_reconstruction,label='Validation',color=RED);ax[1].set_yscale('log');ax[1].set_xlabel('CVAE epoch');ax[1].set_ylabel('Reconstruction MSE (log axis)');ax[1].legend();save(fig,'09_sensitivity')
 log('publication figures complete')
if __name__=='__main__':main()
