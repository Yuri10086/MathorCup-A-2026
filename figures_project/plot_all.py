#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MathorCup A: reproducible publication figures from the frozen result tables.

AI-assisted visualization code. No models are retrained; no numbers are hard-coded
for graphical results. Only the overview process diagram is intentionally omitted.

Usage:
  python plot_all.py --project ./support --output ./publication_figures
  python plot_all.py --project D:/MathorCup2026_A --output D:/MathorCup2026_A/paper_figures

The `--project` root may either be the supplied support folder containing
outputs/rebuild/..., or the existing project root with those same paths.
"""
from __future__ import annotations
import argparse
import json
import math
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
import networkx as nx

NAVY='#18344A'; BLUE='#386A91'; TEAL='#168A81'; LIGHT='#75B9AF'
CORAL='#CF6753'; ORANGE='#E7A65D'; GRAY='#8596A6'; INK='#253342'
MUTED='#6F7C88'; GRID='#DFE5EA'; BG='#FFFFFF'
COL_M={'Mean':GRAY,'Ridge':'#8EA7B4','PLS':'#B0BEC8','RF':BLUE,
       'ET':TEAL,'HGB':'#5A91A9','MLP':ORANGE,'Hierarchy':'#C98D79',
       'HierarchyRater':CORAL,'InnerSelected':'#49788E'}
COL_Q2={'Spearman':GRAY,'Hybrid2plus1':TEAL,'Forward3':BLUE,
        'Redundancy':'#B182A2','TrainSelected':CORAL}
COL_G={'PCA_Gaussian':GRAY,'CopulaIndependent':BLUE,
       'CopulaJoint':TEAL,'CVAE':CORAL}
COL_T={'Unconstrained':GRAY,'PointConstraint':BLUE,
       'UncertaintyConstraint':TEAL,'Reject':CORAL}
LAB_Q2={'Spearman':'Spearman','Hybrid2plus1':'2+1 条件增益',
        'Forward3':'完整前向','Redundancy':'冗余约束','TrainSelected':'训练内选法'}
LAB_G={'PCA_Gaussian':'PCA-Gaussian','CopulaIndependent':'独立 Copula',
       'CopulaJoint':'联合 Copula','CVAE':'CVAE'}
LAB_T={'Unconstrained':'无约束','PointConstraint':'点预测约束',
       'UncertaintyConstraint':'不确定性约束','Reject':'拒绝迁移'}

class Builder:
    def __init__(self, root:Path, out:Path, dpi:int=260):
        self.root=root;self.base=root/'outputs'/'rebuild';self.out=out
        self.out.mkdir(parents=True,exist_ok=True);self.dpi=dpi
        self.status=[];self.source_map={};self.configure()
    def configure(self):
        import matplotlib.font_manager as fm
        available={f.name for f in fm.fontManager.ttflist}
        chinese=['Microsoft YaHei','SimHei','PingFang SC','WenQuanYi Micro Hei']
        # Register the bundled CJK font because Matplotlib may not index TTC collections by default.
        ttc=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        if ttc.exists():
            fm.fontManager.addfont(str(ttc))
            selected=fm.FontProperties(fname=str(ttc)).get_name()
        else:
            selected=next((x for x in chinese if x in available),'DejaVu Sans')
        mpl.rcParams.update({
           'font.family':'sans-serif','font.sans-serif':[selected,'DejaVu Sans','Arial'],
           'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,
           'svg.fonttype':'none','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
           'xtick.labelsize':8,'ytick.labelsize':8,'figure.facecolor':'white',
           'axes.facecolor':'white','axes.edgecolor':'#B5C0C8','axes.linewidth':.7,
           'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'normal',
           'axes.titlepad':10,'grid.color':GRID,'grid.linewidth':.65,
           'savefig.facecolor':'white','legend.frameon':False,
        })
    def read(self, rel):
        p=self.base/rel
        if not p.exists(): raise FileNotFoundError(f'缺少原始结果文件: {p}')
        return pd.read_csv(p)
    def raw(self):
        for p in [self.root/'1.csv',self.root/'1(1).csv',self.root/'assets'/'1.csv',self.root/'data'/'raw'/'1.csv',self.root/'data'/'raw'/'1(1).csv']:
            if p.exists():return pd.read_csv(p,low_memory=False)
        raise FileNotFoundError('风味画像或化学变化图需要原始官方1.csv（放在项目根目录或assets/1.csv）')
    def save(self,fig,name,sources,notes=''):
        fig.savefig(self.out/(name+'.png'),dpi=self.dpi,bbox_inches='tight',pad_inches=.14)
        fig.savefig(self.out/(name+'.pdf'),bbox_inches='tight',pad_inches=.14)
        fig.savefig(self.out/(name+'.svg'),bbox_inches='tight',pad_inches=.14)
        plt.close(fig)
        self.status.append({'figure':name,'sources':sources,'notes':notes})
        print('  ✓',name,flush=True)
    def panel(self,ax,letter,title):
        ax.set_title(f'({letter})  {title}',loc='left',color=NAVY,weight='normal',pad=12)
    def grid(self,ax,axis='y'):
        ax.grid(axis=axis,color=GRID,alpha=.85,lw=.7);ax.set_axisbelow(True)
    def labelbar(self,ax,rects,fmt='{:.3f}',dx=0):
        for r in rects:
            h=r.get_height()
            if np.isfinite(h):ax.annotate(fmt.format(h),(r.get_x()+r.get_width()/2,h),
                (dx,4),textcoords='offset points',ha='center',va='bottom',fontsize=7,color=INK)

    # 图2：数据分布与最低值
    def f02_data(self):
        s=self.read('audit/styles.csv').sort_values('n')
        f=self.read('audit/features.csv').sort_values('minimum_fraction',ascending=False)
        f=f[f.minimum_fraction>0.45].head(12)
        fig=plt.figure(figsize=(12.3,5.5));gs=GridSpec(1,2,figure=fig,wspace=.42,width_ratios=[1.22,1])
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1])
        bar=a.barh(np.arange(len(s)),s.n,color=[TEAL if i in ('Tripel','Faro','Kriek','Lambic') else '#98ABBB' for i in s['style']]);a.set_yticks(range(len(s)),s['style'],fontsize=7.4);a.set_xlim(0,max(s.n)*1.15)
        self.panel(a,'a','22种啤酒风格的样本量');a.set_xlabel('样本数量（款）');a.axvline(5,color=CORAL,ls='--',lw=.9);self.grid(a,'x')
        for p,n in zip(bar,s.n):a.text(n+.3,p.get_y()+p.get_height()/2,str(int(n)),ha='left',va='center',fontsize=7)
        b.barh(np.arange(len(f)),f.minimum_fraction*100,color=[CORAL if v>=.7 else BLUE for v in f.minimum_fraction]);b.invert_yaxis();b.set_yticks(range(len(f)),f.feature,fontsize=7.5);b.set_xlim(0,104)
        self.panel(b,'b','高频最低观测值的化学变量');b.set_xlabel('记录为该变量最小值的样本占比（%）');self.grid(b,'x')
        for i,r in enumerate(f.itertuples()):b.text(r.minimum_fraction*100+1.1,i,f'{r.minimum_fraction:.1%}',va='center',fontsize=7)
        fig.subplots_adjust(left=.12,right=.98,top=.91,bottom=.13)
        self.save(fig,'02_数据审计_风格与最低值',['audit/styles.csv','audit/features.csv'], '最低观测值不自动等于仪器检出限')

    # 图3：Q1模型和逐感官误差
    def f03_q1_compare(self):
        d=self.read('q1/repeat_metrics.csv');pt=self.read('q1/per_target.csv')
        q=d.groupby('model').MAE.agg(['mean','std']).sort_values('mean',ascending=False)
        fig=plt.figure(figsize=(12.5,4.9));gs=GridSpec(1,2,figure=fig,width_ratios=[1.15,1],wspace=.32)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);y=np.arange(len(q))
        a.barh(y,q['mean'],xerr=q['std'],height=.68,color=[COL_M.get(k,GRAY) for k in q.index],ecolor=INK,capsize=2,error_kw={'lw':.85})
        a.set_yticks(y,q.index);a.set_xlabel('50项感官 MAE（0–6分，越低越好）');a.set_xlim(0,q['mean'].max()*1.17);self.grid(a,'x');self.panel(a,'a','各模型两轮交叉验证性能')
        for i,x in enumerate(q['mean']):a.text(x+.008,i,f'{x:.4f}',va='center',fontsize=7.8)
        p=pt[pt.model.isin(['ET','HierarchyRater','MLP','Mean'])]
        p=p.pivot(index='target',columns='model',values='MAE')
        for model in ['Mean','MLP','HierarchyRater','ET']:
            if model in p.columns:
                a0=p[model].dropna().values
                b.boxplot([a0],positions=[[ 'Mean','MLP','HierarchyRater','ET'].index(model)],widths=.55,
                    patch_artist=True,showfliers=False,boxprops={'facecolor':COL_M[model],'edgecolor':'white','alpha':.85},
                    medianprops={'color':'#FFFFFF','lw':1.3},whiskerprops={'color':MUTED},capprops={'color':MUTED})
        b.set_xticks(range(4),['均值基线','MLP','层次评分员网络','ExtraTrees'],rotation=20,ha='right');b.set_ylabel('逐感官维度 MAE（50维分布）');self.panel(b,'b','不同模型的感官维度误差分布');self.grid(b)
        fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.19)
        self.save(fig,'03_Q1_模型性能与维度差异',['q1/repeat_metrics.csv','q1/per_target.csv'],'两轮交叉验证均值，误差条为轮间标准差')

    def f04_q1_masking(self):
        d=self.read('q1/sensitivity.csv')
        q=d[d.experiment.str.contains('mask',case=False)].copy()
        q['ratio']=pd.to_numeric(q.experiment.str.extract(r'mask[_-]([0-9.]+)')[0],errors='coerce')
        q=q.dropna(subset=['ratio']);base=d[d.experiment=='ET_equal_rater']
        aggregate=q.groupby('ratio').MAE.agg(['mean','std'])
        a0=base.MAE.mean();fig,axes=plt.subplots(1,2,figsize=(11.6,4.0),gridspec_kw={'width_ratios':[1.25,1]})
        x=np.r_[0,aggregate.index.values]*100;ym=np.r_[a0,aggregate['mean'].values]
        axes[0].plot(x,ym,color=TEAL,lw=2,marker='o',ms=6)
        axes[0].fill_between(aggregate.index.values*100,aggregate['mean']-aggregate['std'],aggregate['mean']+aggregate['std'],color=TEAL,alpha=.12)
        for i,(xx,yy) in enumerate(zip(x,ym)):axes[0].text(xx,yy+.0018,f'{yy:.4f}',ha='center',fontsize=8)
        axes[0].set_xlabel('额外随机遮蔽化学特征比例（%）');axes[0].set_ylabel('MAE（分）');axes[0].set_xticks(x);self.grid(axes[0]);self.panel(axes[0],'a','随机特征遮蔽下的预测误差')
        ex=d.groupby('experiment').MAE.mean().sort_values()
        rest=ex[~ex.index.str.contains('mask',case=False)].head(8)
        bars=axes[1].barh(np.arange(len(rest)),rest.values,color=[TEAL if z=='ET_equal_rater' else '#8FA6B4' for z in rest.index]);axes[1].set_yticks(range(len(rest)),rest.index,fontsize=7.5);axes[1].invert_yaxis();axes[1].set_xlim(0,max(rest.values)*1.17);axes[1].set_xlabel('MAE（注意比较目标维度）');self.panel(axes[1],'b','评分机制相关敏感性设置');self.grid(axes[1],'x')
        for rect,val in zip(bars,rest.values):axes[1].text(val+.006,rect.get_y()+rect.get_height()/2,f'{val:.4f}',va='center',fontsize=7)
        fig.tight_layout(pad=2.3)
        self.save(fig,'04_Q1_特征遮蔽与标签敏感性',['q1/sensitivity.csv'],'不同目标维度的敏感性行不能直接解释成模型改善')

    # 图5：化合物网络：正负边 + 社区
    def f05_network(self):
        d=self.read('q2/signed_edges.csv');G=nx.Graph()
        for row in d.itertuples():G.add_edge(str(row.source),str(row.target),weight=abs(row.rho),rho=row.rho)
        positive=nx.Graph(((u,v,dat) for u,v,dat in G.edges(data=True) if dat['rho']>0))
        com=list(nx.algorithms.community.greedy_modularity_communities(positive,weight='weight')) if positive.number_of_edges() else []
        membership={n:i for i,s in enumerate(com) for n in s}
        colorset=[TEAL,BLUE,'#8BA5AA','#B69B70','#7F99C4','#8A839F','#A9BBAA']
        node_color=[colorset[membership.get(n,0)%len(colorset)] for n in G.nodes()]
        fig=plt.figure(figsize=(12.0,5.5));gs=GridSpec(1,2,width_ratios=[1.4,1],wspace=.15,figure=fig)
        ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1])
        pos=nx.spring_layout(G,seed=2026,weight='weight',iterations=100)
        pos_e=[(u,v) for u,v,d in G.edges(data=True) if d['rho']>=0]
        neg_e=[(u,v) for u,v,d in G.edges(data=True) if d['rho']<0]
        nx.draw_networkx_edges(G,pos,edgelist=pos_e,edge_color='#95BDB8',alpha=.27,width=.75,ax=ax)
        nx.draw_networkx_edges(G,pos,edgelist=neg_e,edge_color=CORAL,alpha=.44,width=.83,style='dashed',ax=ax)
        nx.draw_networkx_nodes(G,pos,node_size=[10+G.degree[n]*5 for n in G.nodes()],node_color=node_color,edgecolors='white',linewidths=.35,alpha=.94,ax=ax)
        labels=sorted(G.nodes,key=lambda z:G.degree[z],reverse=True)[:7]
        nx.draw_networkx_labels(G,pos,{k:k.replace('_','\n') for k in labels},font_size=6.8,font_color=INK,ax=ax)
        ax.axis('off');self.panel(ax,'a','阈值筛选后的有符号关联活跃网络')
        deg=sorted(G.degree,key=lambda x:x[1],reverse=True)[:12]
        bx.barh(np.arange(len(deg)),[z[1] for z in deg],color=TEAL,height=.7);bx.set_yticks(range(len(deg)),[z[0] for z in deg],fontsize=7);bx.invert_yaxis();bx.set_xlabel('连接数（degree）');self.grid(bx,'x');self.panel(bx,'b','主要高连接度化合物')
        fig.legend(handles=[mpl.lines.Line2D([0],[0],color='#95BDB8',label='正相关'),mpl.lines.Line2D([0],[0],color=CORAL,ls='--',label='负相关')],loc='lower left',bbox_to_anchor=(.11,.02),ncol=2,fontsize=8)
        fig.subplots_adjust(left=.03,right=.985,top=.89,bottom=.11)
        self.save(fig,'05_Q2_有符号化合物网络',['q2/signed_edges.csv'],'节点社区为正相关边上的贪心模块划分，仅用于可视化，非化学家族认证')

    def f06_q2_methods(self):
        d=self.read('closed/q2/summary_metrics.csv').set_index('method')
        st=self.read('closed/q2/stability_summary.csv').set_index('method')
        order=['Spearman','Hybrid2plus1','Forward3','Redundancy','TrainSelected']
        fig,ax=plt.subplots(1,3,figsize=(13.4,4.4),gridspec_kw={'width_ratios':[1.1,1.05,.95]})
        x=np.arange(len(order));c=[COL_Q2[z] for z in order]
        v=d.loc[order,'MAE_mean'].values;sd=d.loc[order,'MAE_std'].values
        bar=ax[0].bar(x,v,yerr=sd,capsize=2.5,color=c,width=.65,error_kw={'lw':.75});ax[0].set_xticks(x,[LAB_Q2[m] for m in order],rotation=32,ha='right');ax[0].set_ylabel('50维MAE（分）');ax[0].set_ylim(0,max(v)*1.20);self.grid(ax[0]);self.panel(ax[0],'a','三特征预测性能')
        self.labelbar(ax[0],bar)
        bw=.36;ax[1].bar(x-bw/2,st.loc[order,'compound_jaccard'],bw,color=BLUE,label='具体化合物');ax[1].bar(x+bw/2,st.loc[order,'group_jaccard'],bw,color=TEAL,label='替代组');ax[1].set_xticks(x,[LAB_Q2[m] for m in order],rotation=32,ha='right');ax[1].set_ylabel('Bootstrap Jaccard');self.grid(ax[1]);ax[1].legend(loc='upper right',fontsize=7.2);self.panel(ax[1],'b','排名稳定性')
        bar2=ax[2].bar(x,st.loc[order,'within_top3_abs_rho'],color=c,width=.63);ax[2].set_xticks(x,[LAB_Q2[m] for m in order],rotation=32,ha='right');ax[2].set_ylabel('Top-3内部平均绝对相关');self.grid(ax[2]);self.panel(ax[2],'c','入选化合物信息冗余')
        self.labelbar(ax[2],bar2,'{:.2f}')
        fig.tight_layout(pad=2.0)
        self.save(fig,'06_Q2_误差稳定性与冗余',['closed/q2/summary_metrics.csv','closed/q2/stability_summary.csv'],'固定2+1与训练内自动选法须分开解读')

    # 图7：四模型生成比较
    def f07_generation_compare(self):
        d=self.read('closed/generation/macro_summary.csv').set_index('model')
        models=['PCA_Gaussian','CopulaIndependent','CopulaJoint','CVAE']
        plots=[('SWD_mean','SWD_std','逐风格SWD ↓'),('style_accuracy_mean','style_accuracy_std','风格分类一致率 ↑'),('joint_floor_MAE_mean','joint_floor_MAE_std','联合低值概率误差 ↓'),('diversity_mean','diversity_std','生成多样性 ↑'),('correlation_MAE_mean','correlation_MAE_std','相关结构误差 ↓'),('floor_mass_MAE_mean','floor_mass_MAE_std','单变量最低值误差 ↓')]
        fig,ax=plt.subplots(2,3,figsize=(13.5,6.6));
        for i,(col,err,title) in enumerate(plots):
            a=ax.flat[i];v=d.loc[models,col];e=d.loc[models,err].fillna(0);rects=a.bar(range(4),v.values,yerr=e.values,color=[COL_G[z] for z in models],width=.64,capsize=2.5,error_kw={'lw':.9});a.set_xticks(range(4),[LAB_G[z] for z in models],rotation=18,ha='right',fontsize=7.4);a.set_ylim(0,(v+e).max()*1.22);self.panel(a,chr(97+i),title);self.grid(a)
            for r,h in zip(rects,v.values):a.annotate(f'{h:.3f}',(r.get_x()+r.get_width()/2,r.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
        fig.tight_layout(pad=2.0)
        self.save(fig,'07_Q3_生成模型六维对照',['closed/generation/macro_summary.csv'],'不同指标不要合成为无依据的单个质量分数')

    # 图8：观测及生成的代表性感官画像
    def f08_portraits(self):
        z=self.read('q3/generated_sensory50.csv');obs=self.read('audit/equal_rater_targets.csv');raw=self.raw();
        if len(obs)!=len(raw): raise ValueError('原始样本与评分员等权目标行数不一致，不能按行序配对')
        obs=obs.copy();obs['style']=raw['tasting_category_fine'].values
        cols=['A_malt_all','A_hops_all','A_esters_all','F_malt_all','F_hops_all','F_esters_all','bitternes','sweetness','acidity','body','aftertaste','overall']
        labels=['麦芽香','酒花香','酯香','麦芽味','酒花味','酯味','苦度','甜度','酸度','酒体','余味','总体']
        styles=['Tripel','Kriek','Lambic'];fig,axs=plt.subplots(1,3,figsize=(14.0,4.9),sharey=True)
        for i,(style,a) in enumerate(zip(styles,axs)):
            p0=obs[obs['style']==style][cols].mean();p1=z[z['style']==style][cols].mean();x=np.arange(len(cols));bw=.35
            a.bar(x-bw/2,p0,bw,color=BLUE,label='历史品评均值');a.bar(x+bw/2,p1,bw,color=TEAL,label='生成谱代理画像');a.set_xticks(x,labels,rotation=55,ha='right',fontsize=7.1);a.set_ylim(0,6.15);self.grid(a);self.panel(a,chr(97+i),f'{style}（历史n={len(obs[obs["style"]==style])}）')
        axs[0].set_ylabel('0–6分感官强度');fig.legend(['历史品评均值','生成谱代理画像'],loc='upper center',bbox_to_anchor=(.5,1.03),ncol=2)
        fig.tight_layout(rect=[0,0,1,.95])
        self.save(fig,'08_Q3_三风格感官画像',['q3/generated_sensory50.csv','audit/equal_rater_targets.csv','官方1.csv'],'生成画像由预测器产生，不是生成啤酒的实测盲评分')

    # 图9：逐风格有效性支持 + SWD
    def f09_perstyle(self):
        cov=self.read('closed/generation/style_coverage.csv').sort_values('n_total',ascending=False)
        m=self.read('closed/generation/metrics_22styles_summary.csv')
        style_order=cov['style'].tolist();models=['PCA_Gaussian','CopulaIndependent','CopulaJoint','CVAE']
        mat=m.pivot(index='style',columns='model',values='SWD_mean').reindex(index=style_order,columns=models)
        fig=plt.figure(figsize=(12.2,7.4));gs=GridSpec(1,3,figure=fig,width_ratios=[.65,1.35,.65],wspace=.26)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[0,2]);y=np.arange(len(style_order))
        a.barh(y,cov.n_total,color='#97AAB9');a.barh(y,cov.n_unique_test,color=TEAL,label='留出');a.set_yticks(y,style_order,fontsize=7.3);a.invert_yaxis();self.panel(a,'a','总样本 / 留出样本');a.set_xlabel('样本量');self.grid(a,'x');a.legend(fontsize=7)
        cm=mpl.colormaps['YlGnBu'].copy();cm.set_bad('#DFE4E8');vals=mat.values
        im=b.imshow(np.ma.masked_invalid(vals),aspect='auto',cmap=cm,interpolation='none',vmin=0,vmax=np.nanpercentile(vals,95))
        b.set_yticks(y,[]);b.set_xticks(range(4),[LAB_G[x] for x in models],rotation=28,ha='right',fontsize=7);self.panel(b,'b','不同风格的条件SWD（灰色=NA）');fig.colorbar(im,ax=b,shrink=.62,label='SWD ↓')
        c.barh(y,cov.n_correlation_contexts,color=BLUE);c.set_yticks(y,[]);c.set_xlim(0,max(4,cov.n_correlation_contexts.max()+1));self.panel(c,'c','可评价相关结构的划分数');c.set_xlabel('有效划分数量');self.grid(c,'x')
        fig.subplots_adjust(left=.1,right=.98,top=.9,bottom=.12)
        self.save(fig,'09_Q3_逐风格评估支持范围',['closed/generation/style_coverage.csv','closed/generation/metrics_22styles_summary.csv'],'无留出样本的风格保持NA，不填零')

    # 图10：联合依赖和低端结构
    def f10_structure(self):
        d=self.read('closed/generation/metrics_22styles_summary.csv')
        cov=self.read('closed/generation/style_coverage.csv')
        styles=cov[cov.n_correlation_contexts>0]['style'].tolist()
        models=['PCA_Gaussian','CopulaIndependent','CopulaJoint','CVAE']
        fig=plt.figure(figsize=(12.8,5.0));gs=GridSpec(1,2,figure=fig,width_ratios=[.9,1.3],wspace=.29)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);
        keys=['joint_floor_MAE_mean','correlation_MAE_mean'];titles=['共同最低值误差','相关结构误差']
        for metric,title in zip(keys,titles):
            pv=d.pivot(index='style',columns='model',values=metric)
            vals=[pv.get(model,pd.Series(dtype=float)).dropna().mean() for model in models]
            row=keys.index(metric)
            x=np.arange(4)+(row-.5)*.32
            a.bar(x,vals,width=.3,color=[COL_G[v] for v in models],alpha=.95 if row==0 else .55,label=title)
        a.set_xticks(np.arange(4),[LAB_G[k] for k in models],rotation=20,ha='right');a.set_ylabel('误差（↓，不同指标不应直接比较大小）');a.legend(fontsize=7);self.grid(a);self.panel(a,'a','两类联合结构误差的模型对照')
        c=d[d['style'].isin(styles)].pivot(index='style',columns='model',values='correlation_MAE_mean').reindex(index=styles,columns=models)
        im=b.imshow(np.ma.masked_invalid(c.values),aspect='auto',cmap='YlGnBu',vmin=0,vmax=np.nanmax(c.values)*1.07)
        b.set_yticks(range(len(c.index)),c.index);b.set_xticks(range(4),[LAB_G[k] for k in models],rotation=20,ha='right');self.panel(b,'b','具备相关结构支持的风格');fig.colorbar(im,ax=b,shrink=.76,label='相关系数MAE')
        for i in range(len(c.index)):
            for j in range(4):
                if np.isfinite(c.values[i,j]):b.text(j,i,f'{c.values[i,j]:.2f}',ha='center',va='center',fontsize=8,color=INK)
        fig.tight_layout(pad=2.0)
        self.save(fig,'10_Q3_联合低值与相关结构',['closed/generation/metrics_22styles_summary.csv','closed/generation/style_coverage.csv'],'结构指标只对有足够留出样本的风格解释')

    # 图11：9酒配对收益风险
    def f11_paired_migration(self):
        d=self.read('closed/migration/per_beer.csv');order=sorted(d.beer.unique());policies=['Unconstrained','PointConstraint','UncertaintyConstraint','Reject']
        fig,axes=plt.subplots(1,2,figsize=(12.2,4.5),sharex=True)
        for pol in policies:
            t=d[d.policy==pol].set_index('beer').reindex(order)
            axes[0].plot(range(len(order)),t.target_gain.values,marker='o',ms=4.5,lw=1.5,label=LAB_T[pol],color=COL_T[pol])
            axes[1].plot(range(len(order)),t.stale_change.values,marker='o',ms=4.5,lw=1.5,label=LAB_T[pol],color=COL_T[pol])
        for a in axes:
            a.axhline(0,color='#4C5965',lw=.9,ls='--');a.set_xticks(range(len(order)),[str(x) for x in order]);a.set_xlabel('留出源啤酒编号');self.grid(a)
        axes[0].set_ylabel('目标风格评价分数变化');axes[1].set_ylabel('独立stale_hops代理变化');self.panel(axes[0],'a','迁移后靠近Tripel的程度');self.panel(axes[1],'b','独立代理记录的风险变化')
        fig.legend(loc='upper center',ncol=4,bbox_to_anchor=(.5,1.05),fontsize=8)
        fig.tight_layout(rect=[0,0,1,.94])
        self.save(fig,'11_Q3_九酒迁移配对效果',['closed/migration/per_beer.csv'],'正值风险变化代表独立代理评分上升，源酒统计单位n=9')

    # 图12：迁移收益-风险-修改-拒绝率
    def f12_risk_tradeoff(self):
        d=self.read('closed/migration/accepted_and_all.csv').set_index('policy');p=['Unconstrained','PointConstraint','UncertaintyConstraint','Reject']
        metrics=[('all_beer_target_gain','目标风格增益 ↑'),('all_beer_risk_rate','独立代理上升比例 ↓'),('rejection_rate','拒绝率 ↓')]
        fig,axs=plt.subplots(1,3,figsize=(12.8,4.1))
        for i,(m,t) in enumerate(metrics):
            a=axs[i];v=d.loc[p,m].values;rects=a.bar(np.arange(4),v,color=[COL_T[x] for x in p],width=.66);a.set_xticks(range(4),[LAB_T[x] for x in p],rotation=27,ha='right',fontsize=7.5);a.set_ylim(0,max(v)*1.30+.015);self.panel(a,chr(97+i),t);self.grid(a)
            for rect,val in zip(rects,v):a.text(rect.get_x()+rect.get_width()/2,rect.get_height()+max(v)*.025+.004,f'{val:.3f}',ha='center',fontsize=7)
        fig.tight_layout(pad=2)
        self.save(fig,'12_Q3_迁移收益风险与拒绝率',['closed/migration/accepted_and_all.csv'],'同一独立代理的风险率并非真实盲评安全率')

    # 额外附录图A：Q1准确-预测图，按啤酒聚合，避免将50维当独立样本
    def sa01_q1_scatter(self):
        p=self.read('q1/continuous_oof.csv');truth=self.read('audit/equal_rater_targets.csv')
        joined=p[['beer']].merge(truth,on='beer',validate='one_to_one')
        cols=[c for c in p.columns if c in truth.columns and c!='beer']
        x=joined[cols].to_numpy(float);y=p[cols].to_numpy(float)
        mask=np.isfinite(x)&np.isfinite(y)
        xx=x[mask];yy=y[mask];fig,ax=plt.subplots(figsize=(6.0,5.3))
        hb=ax.hexbin(xx,yy,gridsize=42,mincnt=1,cmap='YlGnBu',bins='log',extent=(0,6,0,6))
        ax.plot([0,6],[0,6],color=CORAL,ls='--',lw=1.4,label='理想预测');ax.set(xlim=(0,6),ylim=(0,6),xlabel='评分员等权参考均分（0–6）',ylabel='折外模型预测（0–6）');self.panel(ax,'a','ExtraTrees：真实均分与折外预测');fig.colorbar(hb,ax=ax,label='组合频次（log）');ax.legend(loc='upper left');ax.set_aspect('equal');fig.tight_layout()
        self.save(fig,'S01_Q1_评分真实预测密度',['q1/continuous_oof.csv','audit/equal_rater_targets.csv'],'散点为多个目标×多个啤酒，不可将其视为统计独立的12500个啤酒')

    def sa02_q1_targets(self):
        d=self.read('q1/per_target.csv');pv=d[d.model.isin(['ET','MLP','HierarchyRater'])].pivot(index='target',columns='model',values='MAE')
        pv=pv.sort_values('ET',ascending=False)
        fig,ax=plt.subplots(figsize=(8.8,11));im=ax.imshow(pv.values,aspect='auto',cmap='YlGnBu',vmin=0,vmax=np.nanpercentile(pv.values,97))
        ax.set_yticks(range(len(pv)),pv.index,fontsize=6.8);ax.set_xticks(range(len(pv.columns)),pv.columns);self.panel(ax,'a','50项感官属性预测误差（MAE）');fig.colorbar(im,ax=ax,shrink=.45,label='MAE（分）');fig.tight_layout()
        self.save(fig,'S02_Q1_50感官误差热图',['q1/per_target.csv'])

    def sa03_q2_top3(self):
        d=self.read('closed/q2/top3_50x3.csv').sort_values(['target','rank'])
        targets=list(d.target.drop_duplicates());fig,ax=plt.subplots(figsize=(12.5,12.0));v=d.pivot(index='target',columns='rank',values='bootstrap_frequency').reindex(targets)
        im=ax.imshow(v.values,aspect='auto',cmap='YlGnBu',vmin=0,vmax=1)
        ax.set_yticks(range(len(targets)),targets,fontsize=6.7);ax.set_xticks(range(3),['Top-1','Top-2','Top-3']);self.panel(ax,'a','50项感官Top-3入选频率（标签为实际化学物质）')
        p=d.pivot(index='target',columns='rank',values='chemical').reindex(targets)
        for i in range(len(targets)):
            for j in range(3):
                z=str(p.iloc[i,j]);z=z[:23]+'…' if len(z)>24 else z
                ax.text(j,i,z,ha='center',va='center',fontsize=6.2,color=INK)
        fig.colorbar(im,ax=ax,shrink=.38,label='100次Bootstrap入选比例');fig.tight_layout()
        self.save(fig,'S03_Q2_150化合物稳定性总图',['closed/q2/top3_50x3.csv'],'文本密集，建议放附录或单独横向页面')

    def sa04_q3_training(self):
        d=self.read('closed/generation/training_curves.csv')
        fig,axes=plt.subplots(1,2,figsize=(10.6,3.8))
        cols=[('reconstruction','训练重构损失'),('validation_reconstruction','验证重构损失')]
        for a,(col,title) in zip(axes,cols):
            g=d.groupby('epoch')[col].agg(['mean','std']).fillna(0)
            a.plot(g.index,g['mean'],lw=1.9,color=TEAL if col=='reconstruction' else CORAL)
            a.fill_between(g.index,(g['mean']-g['std']).to_numpy(),(g['mean']+g['std']).to_numpy(),alpha=.14,color=TEAL if col=='reconstruction' else CORAL)
            a.set(xlabel='Epoch',ylabel='重构损失')
            self.grid(a);self.panel(a,'a' if col=='reconstruction' else 'b',title)
        fig.tight_layout(pad=2.1)
        self.save(fig,'S04_Q3_生成网络训练曲线',['closed/generation/training_curves.csv'],'验证重构损失不是最终SWD指标')

    def sa05_q3_chemical_change(self):
        d=self.read('closed/migration/spectra.csv');raw=self.raw();beers=sorted(d.beer.unique());beer=beers[0]
        base=raw[raw.beer==beer]
        if len(base)!=1:raise ValueError('源酒编号在官方1.csv无法唯一匹配')
        features=[c for c in raw.columns if c in d.columns and pd.api.types.is_numeric_dtype(raw[c]) and c not in ['beer','beer_id']]
        t=d[d.beer==beer].set_index('policy');pols=['Unconstrained','PointConstraint','UncertaintyConstraint','Reject']
        if len(t)!=len(pols):raise ValueError('迁移策略不完整')
        std=raw[features].std(axis=0).replace(0,np.nan)
        delta=t.loc[pols,features].subtract(base.iloc[0][features].astype(float),axis=1).divide(std,axis=1)
        ds=delta.abs().mean().sort_values(ascending=False);chosen=list(ds.head(12).index)
        v=delta[chosen].to_numpy(float);vmax=max(.1,np.nanpercentile(abs(v),95))
        fig,ax=plt.subplots(figsize=(11.5,3.7));im=ax.imshow(v,cmap='RdBu_r',aspect='auto',vmin=-vmax,vmax=vmax);ax.set_yticks(range(4),[LAB_T[x] for x in pols]);ax.set_xticks(range(len(chosen)),chosen,rotation=35,ha='right',fontsize=7.2)
        self.panel(ax,'a',f'固定源酒 {beer}：各策略化学变化（按历史总体SD缩放）');fig.colorbar(im,ax=ax,label='相对原酒变化 / 总体标准差');fig.tight_layout()
        self.save(fig,'S05_Q3_单酒化学变化',['closed/migration/spectra.csv','官方1.csv'],'固定最小beer ID作为展示案例，非按最优迁移成绩挑选')

    def run(self):
        jobs=[self.f02_data,self.f03_q1_compare,self.f04_q1_masking,self.f05_network,
              self.f06_q2_methods,self.f07_generation_compare,self.f08_portraits,
              self.f09_perstyle,self.f10_structure,self.f11_paired_migration,self.f12_risk_tradeoff,
              self.sa01_q1_scatter,self.sa02_q1_targets,self.sa03_q2_top3,
              self.sa04_q3_training,self.sa05_q3_chemical_change]
        failed=[]
        for f in jobs:
            try: f()
            except Exception as ex:
                failed.append({'figure':f.__name__,'error':f'{type(ex).__name__}: {ex}'})
                print('  ✗',f.__name__,':',ex,flush=True)
        meta={'sources_root':str(self.base),'rendered':len(self.status),'failed':failed,'figures':self.status}
        (self.out/'figure_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        if failed:raise RuntimeError(f'{len(failed)} figures failed; see figure_manifest.json')
        print(f'\n生成完毕：{len(self.status)}张图 × PNG/PDF/SVG，共{len(self.status)*3}个文件。')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(__file__).parent/'support',help='解压后的最终工程根目录')
    ap.add_argument('--output',type=Path,default=Path(__file__).parent/'publication_figures',help='图件输出目录')
    ap.add_argument('--dpi',type=int,default=260)
    args=ap.parse_args();Builder(args.project.resolve(),args.output.resolve(),args.dpi).run()
