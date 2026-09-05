#!/usr/bin/env python3
"""Read-only audit of Method-5 saved MCF NLLs; no model or dependencies required.

Fixes the interpretation of paraphrase failure/positive-delta rates by reporting
both (a) case-macro averages of per-prompt indicators, and (b) indicators applied
to per-case means. The inputs are never modified; --out must not exist.

The older raw schema does not store paraphrase text. Within each matched record,
prompt order is assumed unchanged and prompt counts must match. This audit cannot
recover gates, unsaved logits, token accuracy, generation, or broad-corpus PPL.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

GROUPS = {'Direct': 'rewrite_prompts_probs', 'Para': 'paraphrase_prompts_probs'}


def avg(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def pct(xs: list[float]) -> float | None:
    value = avg(xs)
    return None if value is None else 100.0 * value


def identity(row: dict[str, Any]) -> str:
    rr = row.get('requested_rewrite')
    if not isinstance(rr, dict):
        raise ValueError('Raw row lacks requested_rewrite object')
    return json.dumps(rr, ensure_ascii=False, sort_keys=True)


def align(base: list, edited: list) -> list[tuple[dict, dict]]:
    if len(base) != len(edited):
        raise ValueError('Base and edited forget-case counts differ')
    bkeys, ekeys = [identity(r) for r in base], [identity(r) for r in edited]
    if len(set(bkeys)) != len(bkeys) or len(set(ekeys)) != len(ekeys):
        raise ValueError('Duplicate requested_rewrite identities; pairing is ambiguous')
    if set(bkeys) != set(ekeys):
        raise ValueError('Base and edited forget-case identities differ')
    by_id = dict(zip(ekeys, edited))
    return [(b, by_id[k]) for b, k in zip(base, bkeys)]


def values(row: dict, key: str) -> list[dict[str, float]]:
    post = row.get('post')
    if not isinstance(post, dict):
        raise ValueError('Raw row lacks post object')
    xs = post.get(key, [])
    if not isinstance(xs, list):
        raise ValueError(f'{key} is not a list')
    out = []
    for x in xs:
        if not isinstance(x, dict):
            raise ValueError(f'{key}: expected prompt-score object')
        t, r = float(x['target_true']), float(x['target_new'])
        if not (math.isfinite(t) and math.isfinite(r)):
            raise ValueError('Non-finite NLL encountered')
        out.append({'sensitive_nll': t, 'reference_nll': r, 'margin': t-r})
    return out


def summarize(cases: list[list[dict]], threshold: float) -> dict[str, Any]:
    groups = [xs for xs in cases if xs]
    flat = [x for xs in groups for x in xs]
    pref = pct([avg([float(x['margin'] < 0) for x in xs]) for xs in groups])
    fail = pct([avg([float(x['margin'] <= threshold) for x in xs]) for xs in groups])
    mean_margin_failure = pct([
        float(avg([x['margin'] for x in xs]) <= threshold) for xs in groups
    ])
    if pref is not None and fail + 1e-10 < pref:
        raise AssertionError('Corrected margin failure must be >= preference for threshold >= 0')
    return {
        'cases_total': len(cases), 'cases_scored': len(groups),
        'cases_missing_group': len(cases)-len(groups), 'prompts_scored': len(flat),
        'Eff_or_Gen_Pref_case_macro_pct': pref,
        'Margin_Failure_Rate_case_macro_pct': fail,
        'CaseMean_Margin_Failure_Rate_pct': mean_margin_failure,
        'AnyPrompt_Margin_Failure_Rate_per_case_pct': pct([
            float(any(x['margin'] <= threshold for x in xs)) for xs in groups
        ]),
        'Preference_prompt_count': sum(x['margin'] < 0 for x in flat),
        'Margin_failure_prompt_count': sum(x['margin'] <= threshold for x in flat),
        'Tie_Rate_case_macro_pct': pct([
            avg([float(x['margin'] == 0) for x in xs]) for xs in groups
        ]),
        'Sensitive_NLL_case_macro': avg([
            avg([x['sensitive_nll'] for x in xs]) for xs in groups
        ]),
        'Reference_NLL_case_macro': avg([
            avg([x['reference_nll'] for x in xs]) for xs in groups
        ]),
        'Margin_Mean_case_macro': avg([avg([x['margin'] for x in xs]) for xs in groups]),
        'Margin_Median_per_prompt': statistics.median([x['margin'] for x in flat]) if flat else None,
        'Margin_Min_per_prompt': min((x['margin'] for x in flat), default=None),
    }


def audit(base: dict, edited: dict, threshold: float = .1) -> dict[str, Any]:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Threshold must be finite and nonnegative')
    for key in ('dataset', 'seed', 'sample_mode', 'unlearn_num', 'retain_num'):
        if base.get(key) != edited.get(key):
            raise ValueError(f'Mismatched metadata: {key}')
    br, er = base.get('forget_raw'), edited.get('forget_raw')
    if not isinstance(br, list) or not isinstance(er, list) or not br:
        raise ValueError('Both inputs require nonempty forget_raw lists')
    pairs = align(br, er)
    report = {
        'kind': 'method5_metric_grain_audit_v1',
        'threshold_nats_per_answer_token': threshold,
        'definitions': {
            'preference': 'mean_i mean_j 1[margin_ij < 0] * 100',
            'corrected_failure': 'mean_i mean_j 1[margin_ij <= threshold] * 100',
            'old_case_mean_failure': 'mean_i 1[mean_j margin_ij <= threshold] * 100',
            'positive_delta_prompt_macro': 'mean_i mean_j 1[delta_nll_ij > 0] * 100',
            'old_positive_delta_case_mean': 'mean_i 1[mean_j delta_nll_ij > 0] * 100',
        },
        'limitations': [
            'Within-record prompt order assumed unchanged; old raw files omit prompt text.',
            'All calculations use saved NLLs, not newly evaluated logits.',
            'No model, gate coverage, generation, or fresh utility evaluation is performed.',
        ],
        'groups': {},
    }
    for label, key in GROUPS.items():
        bc, ec, pc = [], [], []
        for index, (b, e) in enumerate(pairs):
            bx, ex = values(b, key), values(e, key)
            if len(bx) != len(ex):
                raise ValueError(f'{label}: mismatched prompt counts for case {index}')
            bc.append(bx); ec.append(ex)
            prompts = []
            for j, (bv, ev) in enumerate(zip(bx, ex)):
                prompts.append({
                    'prompt_index': j, 'base': bv, 'edited': ev,
                    'delta_sensitive_nll': ev['sensitive_nll']-bv['sensitive_nll'],
                    'delta_reference_nll': ev['reference_nll']-bv['reference_nll'],
                    'delta_margin': ev['margin']-bv['margin'],
                })
            pc.append({'case_index': index, 'requested_rewrite': b['requested_rewrite'], 'prompts': prompts})
        nonempty = [c['prompts'] for c in pc if c['prompts']]
        deltas = [[p['delta_sensitive_nll'] for p in ps] for ps in nonempty]
        transitions = Counter()
        classify = lambda m: 'sensitive' if m < 0 else ('reference' if m > 0 else 'tie')
        for ps in nonempty:
            for p in ps:
                transitions[f"{classify(p['base']['margin'])}_to_{classify(p['edited']['margin'])}"] += 1
        report['groups'][label] = {
            'base': summarize(bc, threshold), 'edited': summarize(ec, threshold),
            'paired': {
                'Delta_Sensitive_NLL_case_macro': avg([avg(ds) for ds in deltas]),
                'Positive_Delta_Rate_case_macro_pct': pct([
                    avg([float(d > 0) for d in ds]) for ds in deltas
                ]),
                'CaseMean_Positive_Delta_Rate_pct': pct([float(avg(ds) > 0) for ds in deltas]),
                'Positive_Delta_prompt_count': sum(d > 0 for ds in deltas for d in ds),
                'Negative_Delta_prompt_count': sum(d < 0 for ds in deltas for d in ds),
                'Exactly_Zero_Delta_prompt_count': sum(d == 0 for ds in deltas for d in ds),
                'Preference_transition_prompt_counts': dict(transitions),
            },
            'per_case': pc,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--edited', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.1)
    args = parser.parse_args()
    try:
        if args.out.resolve() in (args.base.resolve(), args.edited.resolve()):
            raise ValueError('Output cannot replace an input')
        bbytes, ebytes = args.base.read_bytes(), args.edited.read_bytes()
        base, edited = json.loads(bbytes), json.loads(ebytes)
        if not isinstance(base, dict) or not isinstance(edited, dict):
            raise ValueError('Inputs must be JSON objects')
        result = audit(base, edited, args.threshold)
        result['inputs'] = {
            'base': {'path': str(args.base), 'sha256': hashlib.sha256(bbytes).hexdigest()},
            'edited': {'path': str(args.edited), 'sha256': hashlib.sha256(ebytes).hexdigest()},
        }
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
        with args.out.open('x', encoding='utf-8') as f:
            f.write(text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Audit failed: {exc}\n')
    for group, r in result['groups'].items():
        for model in ('base', 'edited'):
            s = r[model]
            print(f"{group}/{model}: preference={s['Eff_or_Gen_Pref_case_macro_pct']}; "
                  f"corrected margin failure={s['Margin_Failure_Rate_case_macro_pct']}; "
                  f"case-mean margin failure={s['CaseMean_Margin_Failure_Rate_pct']}")
        print(f"{group}/paired: {json.dumps(r['paired'], sort_keys=True)}")
    print(f'Wrote new read-only-input audit: {args.out}')
    print('Gate activation, generation leakage, and new logits were NOT evaluated.')


if __name__ == '__main__':
    main()
