#!/usr/bin/env python3
"""Compare per-probe RWKU base generations from paper/decomposition or two decompositions.

Run: python compare_rwku_base_evaluations.py BASE_A.json BASE_B.json

Paper format: {"details": {"same_50_efficacy": [...], "neighbors": [...]}}
Decomposition rows format: {"base": {"same50": [...], "neighbors": [...]}}
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
pa, de = map(Path, sys.argv[1:])
paper = json.loads(pa.read_text())
decomp = json.loads(de.read_text())
for file in (pa, de):
    print('file:', file, 'sha256:', hashlib.sha256(file.read_bytes()).hexdigest())
assert 'details' in paper or 'base' in paper, 'First file is not a recognized RWKU per-probe JSON'
assert 'details' in decomp or 'base' in decomp, 'Second file is not a recognized RWKU per-probe JSON'

mapping = {
    'same_50_efficacy': 'same50',
    'heldout_level1': 'heldout_level1',
    'heldout_level2': 'heldout_level2',
    'heldout_level2_paraphrase': 'heldout_paraphrase',
    'neighbors': 'neighbors',
}

def key(x):
    h = str(x.get('source_record_sha256') or '').strip()
    if h:
        return ('sha256', h)
    return ('content', str(x.get('subject')), str(x.get('query')), str(x.get('answer')))

def recovered(x):
    return bool(x.get('recovered', x.get('recovery_success', False)))

for left, right in mapping.items():
    a = paper['details'].get(left) if 'details' in paper else paper['base'].get(right)
    b = decomp['details'].get(left) if 'details' in decomp else decomp['base'].get(right)
    if not isinstance(a, list) or not isinstance(b, list):
        continue
    ka = collections.defaultdict(list)
    kb = collections.defaultdict(list)
    for entry in a: ka[key(entry)].append(entry)
    for entry in b: kb[key(entry)].append(entry)
    different_counts = [(k,len(ka[k]),len(kb[k])) for k in ka.keys()|kb.keys() if len(ka[k]) != len(kb[k])]
    flips, pred_diff = [], []
    for k in ka.keys() & kb.keys():
        for x,y in zip(ka[k],kb[k]):
            if recovered(x) != recovered(y):
                flips.append((k, x.get('query'), x.get('answer'), recovered(x), recovered(y), x.get('prediction'), y.get('prediction')))
            elif (x.get('prediction') or '') != (y.get('prediction') or ''):
                pred_diff.append((k, x.get('query'), x.get('prediction'), y.get('prediction')))
    print('\nGroup:', left, '<->', right)
    print('paper count/recovered:', len(a), sum(recovered(x) for x in a))
    print('decomp count/recovered:', len(b), sum(recovered(x) for x in b))
    print('missing/duplicate-key differences:', len(different_counts))
    for item in different_counts[:8]: print('  key count difference:',item)
    print('matched recovery flips:',len(flips))
    for item in flips[:20]: print('  FLIP:',repr(item))
    print('matched same-recovery but different-text cases:',len(pred_diff))
    for item in pred_diff[:5]: print('  TEXT:',repr(item))
