#!/usr/bin/env python3
"""2026 MathorCup A/Q1: official z-score supervision vs original 0-6 panel labels.
AI-assisted research scaffold (OpenAI GPT-6, 2026-10-27); team must verify,
change as needed, and disclose AI use according to competition rules.

Run: python q1_reference_rebuild.py --data-dir /path/to/files --out-dir /path/to/results
Inputs: official 1.csv and 4.csv; independent original-public-data Raw sensory data anonymized.csv
and scorecard_schema.json (extracted column names/order from official 3.xlsx).
No sample's own sensory labels are used when generating its held-out predictions.
"""
import argparse, csv, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

ALIASES = {'A_ester_fruity': 'A_esters_fruity',
           'F_ester_fruity': 'F_esters_fruity', '4vg': 'X4vg'}
PARENTS = {
 'A_malt_all': ['A_malt_grain','A_malt_bread','A_malt_cara','A_malt_burn'],
 'A_hops_all': ['A_hops_citrus','A_hops_tropical','A_hops_noble','A_hops_woody'],
 'A_esters_all': ['A_esters_ethac','A_esters_isoaa','A_esters_flower','A_esters_fruity'],
 'F_malt_all': ['F_malt_grain','F_malt_bread','F_malt_cara','F_malt_burn'],
 'F_hops_all': ['F_hops_citrus','F_hops_tropical','F_hops_noble','F_hops_woody'],
 'F_esters_all': ['F_esters_ethac','F_esters_isoaa','F_esters_flower','F_esters_fruity']}

def file_for(folder, base):
    a=Path(folder)/base
    if a.is_file(): return a
    stem=Path(base).stem
    hits=sorted(Path(folder).glob(stem+'*'+Path(base).suffix))
    if len(hits)!=1: raise FileNotFoundError(f'Cannot uniquely locate {base}: {hits}')
    return hits[0]

def read_records(p):
    # Raw data file may use cp1252 non-UTF8 characters in comments.
    try:
        with open(p, newline='', encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(p, newline='', encoding='cp1252') as f: rows=list(csv.DictReader(f))
    return [{ALIASES.get(k,k):v for k,v in row.items()} for row in rows]

def val(s):
    if s is None or not str(s).strip() or str(s).strip().lower() in ('na','nan','null'):
        return np.nan
    try:return float(s)
    except ValueError:return np.nan

def output_csv(path, names, ids, vals, label='row_index'):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow([label]+list(names))
        for i,v in zip(ids,vals):w.writerow([i]+[round(float(x),7) for x in v])

def round_score(pred):return np.clip(np.floor(pred+0.5),0,6).astype(int)

def metrics(y,p, names):
    af=np.array([i for i,n in enumerate(names) if n.startswith(('A_','F_'))]);
    disc=round_score(p); truth=round_score(y)
    violations=sum(int((p[:,names.index(ch)]>p[:,names.index(par)]+1e-9).sum())
                   for par,chs in PARENTS.items() for ch in chs)
    return dict(mae_af30=float(np.mean(abs(y[:,af]-p[:,af]))),
                mae_all50=float(np.mean(abs(y-p))),
                rmse_all50=float(np.sqrt(np.mean((y-p)**2))),
                exact_af30=float(np.mean(disc[:,af]==truth[:,af])),
                within1_af30=float(np.mean(abs(disc[:,af]-truth[:,af])<=1)),
                parent_child_violations=violations)

def verify_scorecard(raw, names):
    nonzero=0;bad=0
    for row in raw:
        for parent,chs in PARENTS.items():
            pv=val(row[parent]);assert np.isfinite(pv)
            for child in chs:
                cv=val(row[child]);assert np.isfinite(cv)
                if cv>0:
                    nonzero+=1;bad+=int(cv!=pv)
                bad+=int(cv>pv)
    assert bad==0, f'Scorecard hierarchical relationship violated: {bad}'
    return nonzero

def make_estimator(kind,seed):
    if kind=='ET':
        return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
          ('regress',ExtraTreesRegressor(n_estimators=260,max_features=.7,
                                      min_samples_leaf=2,random_state=seed,n_jobs=2))])
    if kind=='Ridge':
        return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('scale',StandardScaler()),('regress',Ridge(alpha=100.))])
    raise ValueError(kind)

def affine_fit(z,y):
    # learned on TRAIN BEERS ONLY. Targetwise affine calibration; avoid extrapolation extremes
    center=z.mean(axis=0); scale=z.std(axis=0);scale=np.where(scale<1e-8,1.,scale)
    zs=(z-center)/scale
    slope=((zs-zs.mean(axis=0))*(y-y.mean(axis=0))).sum(axis=0)
    var=((zs-zs.mean(axis=0))**2).sum(axis=0)
    slope=np.divide(slope,var,out=np.zeros_like(slope),where=var>1e-12)
    intercept=y.mean(axis=0)-slope*zs.mean(axis=0)
    return center,scale,slope,intercept

def affine_apply(z,m):
    c,s,b,a=m;return np.clip((z-c)/s*b+a,0,6)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--out-dir',default='q1_results')
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    data=Path(args.data_dir)
    schema=json.loads((Path(__file__).parent/'scorecard_schema.json').read_text('utf-8'))
    names=[ALIASES.get(n,n) for n in schema['scorecard'][1:-1]]
    assert len(names)==50 and len(set(names))==50
    chem=read_records(file_for(data,'1.csv'))
    official=read_records(file_for(data,'4.csv'))
    raw=read_records(file_for(data,'Raw sensory data anonymized.csv'))
    assert len(chem)==len(official)==250 and len(raw)==2197
    assert all(a['beer']==b['beer'] for a,b in zip(chem,official))
    features=[k for k in chem[0] if k not in ('beer','beer_id','tasting_category_fine')]
    assert len(features)==231
    X=np.array([[val(r[n]) for n in features] for r in chem],float)
    Z=np.array([[val(r[n]) for n in names] for r in official],float)
    assert np.all(np.isfinite(Z)), 'Expected no NaN in official panel target'
    grouper=defaultdict(list)
    for r in raw: grouper[int(float(r['beer_name']))].append(r)
    assert set(grouper)==set(range(1,251)), 'Ambiguous rowwise anonymized mapping'
    Y=np.array([[np.mean([val(rr[n]) for rr in grouper[k]]) for n in names]
                for k in range(1,251)],float)
    assert np.isfinite(Y).all() and Y.min()>=0 and Y.max()<=6
    raw_nonzero=verify_scorecard(raw,names)
    correlations=np.array([np.corrcoef(Y[:,j],Z[:,j])[0,1] for j in range(50)])
    # Repeatable sanity check for chosen sample mapping against random row correspondences
    rng=np.random.default_rng(42)
    permutation_medians=[]
    for _ in range(100):
        r=rng.permutation(250)
        permutation_medians.append(np.median([np.corrcoef(Y[r,j],Z[:,j])[0,1] for j in range(50)]))
    if min(correlations)<.90 or np.median(correlations)<max(permutation_medians)+.6:
        raise ValueError('Original raw-data beer alignment not strongly supported')
    styles=np.array([r['tasting_category_fine'] for r in chem]);c=defaultdict(int)
    for s in styles:c[s]+=1
    cats=np.array([s if c[s]>=5 else '__RARE__' for s in styles])
    assert min(defaultdict(int,((s,int((cats==s).sum())) for s in set(cats))).values())>=5
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
    methods=['MeanRaw','RidgeRaw','ETRaw','ETOfficialZAffine','RidgeOfficialZAffine']
    pred={k:np.full_like(Y,np.nan) for k in methods}
    pred_z=np.full_like(Z,np.nan)
    fold_marks=np.full(250,-1,int)
    for fold,(tr,va) in enumerate(skf.split(X,cats)):
        fold_marks[va]=fold+1
        pred['MeanRaw'][va]=np.broadcast_to(Y[tr].mean(axis=0),(len(va),50))
        for kind in ('ET','Ridge'):
            er=make_estimator(kind,2026+fold)
            er.fit(X[tr],Y[tr]);pred[kind+'Raw'][va]=np.clip(er.predict(X[va]),0,6)
            ez=make_estimator(kind,2026+fold)
            ez.fit(X[tr],Z[tr]);q=ez.predict(X[va])
            if kind=='ET':pred_z[va]=q
            m=affine_fit(Z[tr],Y[tr])
            pred[kind+'OfficialZAffine'][va]=affine_apply(q,m)
        print('fold',fold+1,'n_train',len(tr),'n_valid',len(va), flush=True)
    assert np.all(fold_marks>0)
    results={k:metrics(Y,v,names) for k,v in pred.items()}
    z_mae=float(np.mean(np.abs(Z-pred_z)))
    rng_boot=np.random.default_rng(2026)
    af=np.array([i for i,n in enumerate(names) if n.startswith(('A_','F_'))])
    et_err=np.abs(Y[:,af]-pred['ETRaw'][:,af]).mean(axis=1)
    pairwise={};
    for m in ('ETOfficialZAffine',):
        delta=np.abs(Y[:,af]-pred[m][:,af]).mean(axis=1)-et_err
        draws=rng_boot.integers(0,250,size=(10000,250))
        boots=delta[draws].mean(axis=1)
        pairwise[m]={'MAE_other_minus_ETRaw':float(delta.mean()),
            'bootstrap_95pct_CI':list(map(float,np.quantile(boots,[.025,.975])))}
    output_csv(out/'reference_raw_panel_means_250.csv',names,range(1,251),Y)
    output_csv(out/'official_standardized_targets_250.csv',names,range(1,251),Z)
    output_csv(out/'official_Z_ET_oof_predictions_250.csv',names,range(1,251),pred_z)
    for k,v in pred.items():
        output_csv(out/(k+'_oof_scores_250.csv'),names,range(1,251),v)
        output_csv(out/(k+'_oof_integer_250.csv'),names,range(1,251),round_score(v))
    # 250 target beer ID, no raw as prediction: only CV predictions, not in-sample fit
    with open(out/'split_assignments.csv','w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['row_index','official_beer','style','fold','n_raw_tastings'])
        for i in range(250):w.writerow([i+1,chem[i]['beer'],styles[i],fold_marks[i],len(grouper[i+1])])
    audit={'source_scope':{
      'official_inputs':['1.csv','3.xlsx(scorecard only)','4.csv(z scores)'],
      'external_reference':'Raw sensory data anonymized.csv (original paper, same 250 beers)',
      'original_paper':'Schreurs et al., Nature Communications (2024), doi:10.1038/s41467-024-46346-0',
      'raw_file_not_official_attachment':True},
      'sample_count':250,'chemical_features':231,'raw_panel_rows':len(raw),
      'panel_group_size_minmax':[min(map(len,grouper.values())),max(map(len,grouper.values()))],
      'nonzero_child_parent_pairs':raw_nonzero,
      'min_median_max_target_corr_Y_vs_Z':list(map(float,[correlations.min(),np.median(correlations),correlations.max()])),
      'max_shuffled_median_correlation':float(max(permutation_medians)),
      'missing_chemical_cells':int(np.isnan(X).sum()),'cv_folds':5,
      'oof_official_z_ET_MAE_50':z_mae,
      'methods':results,'paired_beer_bootstrap_comparisons':pairwise,'label_definition':'beer-level mean of original individual panelist ratings',
      'validation':'beer-level stratified 5fold out-of-fold; all supervised calibration fitted on training beers only',
      'warning':'same 250 beers in original-paper reference; not independent external validation; no access to official private test data; labels must never be presented as chemistry-based predictions'}
    (out/'audit_results.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'alignment_corr_median':audit['min_median_max_target_corr_Y_vs_Z'][1],
                      'official_z_ET_MAE50':z_mae,'model_metrics':results},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
