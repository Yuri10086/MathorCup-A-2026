#!/usr/bin/env python3
"""MathorCup A deliverable preflight (AI-assisted draft, verify independently)."""
import csv, json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
schema=json.loads((root/'q1/scorecard_schema.json').read_text(encoding='utf-8'))
expected=[('A_esters_fruity' if x=='A_ester_fruity' else 'F_esters_fruity' if x=='F_ester_fruity' else 'X4vg' if x=='4vg' else x) for x in schema['scorecard'][1:-1]]
af=[v for v in expected if v.startswith(('A_','F_'))]
parents={'A_malt_all':['A_malt_grain','A_malt_bread','A_malt_cara','A_malt_burn'],
         'A_hops_all':['A_hops_citrus','A_hops_tropical','A_hops_noble','A_hops_woody'],
         'A_esters_all':['A_esters_ethac','A_esters_isoaa','A_esters_flower','A_esters_fruity'],
         'F_malt_all':['F_malt_grain','F_malt_bread','F_malt_cara','F_malt_burn'],
         'F_hops_all':['F_hops_citrus','F_hops_tropical','F_hops_noble','F_hops_woody'],
         'F_esters_all':['F_esters_ethac','F_esters_isoaa','F_esters_flower','F_esters_fruity']}
assert len(af)==30

def read_csv(fname,cols):
    with (root/'results/q1'/fname).open(encoding='utf-8-sig',newline='') as f:
        items=list(csv.DictReader(f))
    assert len(items)==250,(fname,len(items))
    assert list(items[0])[1:]==cols,(fname,'unexpected names')
    assert len(set(r['row_index'] for r in items))==250
    assert [int(r['row_index']) for r in items]==list(range(1,251))
    return items

vals=read_csv('q1_AF30_oof_integer_250.csv',af)
for r in vals:
    assert all(float(r[c]).is_integer() and 0<=float(r[c])<=6 for c in af)
    assert all(float(r[p])>=float(r[ch]) for p,chs in parents.items() for ch in chs)
print('PASS Q1 250 x 30 integers, row order, field names, and hierarchy')

ref=read_csv('reference_raw_panel_means_250.csv',expected)
assert all(0<=float(r[c])<=6 for r in ref for c in expected)
print('PASS reference target 250 x 50; stored separate from OOF')

j=json.loads((root/'results/q1/audit_results.json').read_text(encoding='utf-8'))
assert j['sample_count']==250 and j['chemical_features']==231
assert j['cv_folds']==5
print('PASS provenance, 5-fold audit')

needed={'Q2 Top-3':root/'results/q2/q2_top3_50x3.csv',
        'Q3 generated':root/'results/q3/q3_generated_spectra.csv',
        'Q3 transfer':root/'results/q3/q3_transfer_defects.csv',
        'AI use PDF':root/'submission/AI工具使用详情说明.pdf',
        'Final paper PDF':root/'submission/A队号.pdf'}
for title,path in needed.items():
    found=path.exists() or (title=='Final paper PDF' and bool(list((root/'submission').glob('A*.pdf'))))
    print(('PASS ' if found else 'PENDING ')+title)
print('Note: script validates structure, not statistical correctness of unreviewed findings.')

if (root/'results/q2/q2_top3_50x3.csv').exists():
    with (root/'results/q2/q2_top3_50x3.csv').open(encoding='utf-8-sig',newline='') as f:
        q2=list(csv.DictReader(f))
    assert len(q2)==150 and len(set(r['sensory'] for r in q2))==50
    assert all(len({r['chemical'] for r in q2 if r['sensory']==t})==3 for t in set(r['sensory'] for r in q2))
    print('PASS Q2 structural 50 x 3 candidate chemicals; chemistry validation PENDING')
