#!/usr/bin/env python3
"""Paired qualitative/quantitative analysis of BLIP caption predictions.

Input prediction files: JSON list of {image_id, caption}, or COCO-style
{annotations:[{image_id,caption},...]}. Optional COCO reference annotations.
No third-party Python dependencies are required.
"""
import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

STOP = set('a an the of on in at to with and or for from by is are was were be being been this that these those there it its as into over under near next while two three four one'.split())
CATEGORIES = {
    'people': {'man','woman','person','people','boy','girl','child','children','men','women','baby','player','skier','surfer','rider','crowd'},
    'animals': {'dog','cat','horse','bird','cow','sheep','elephant','bear','zebra','giraffe','animal','puppy'},
    'vehicles': {'car','bus','truck','train','bike','bicycle','motorcycle','boat','airplane','plane','vehicle'},
    'food': {'food','pizza','cake','sandwich','plate','fruit','banana','apple','bread','meal','dish'},
    'sports': {'tennis','baseball','soccer','football','basketball','skateboard','surfboard','ski','ball'},
    'indoor': {'room','kitchen','bedroom','bathroom','table','desk','couch','chair','bed','computer'},
    'outdoor': {'street','road','beach','park','grass','field','mountain','river','lake','ocean','snow','sky'},
}

def tokenize(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())

def load_preds(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(obj, dict):
        obj = obj.get('annotations', obj.get('results', obj.get('predictions')))
    if not isinstance(obj, list):
        raise ValueError(f'{path}: expected list of predictions')
    out = {}
    for item in obj:
        if 'image_id' not in item or 'caption' not in item:
            raise ValueError(f'{path}: prediction must have image_id and caption')
        key = str(item['image_id'])
        if key in out:
            raise ValueError(f'{path}: duplicate image_id {key}')
        out[key] = str(item['caption']).strip()
    return out

def load_refs(path):
    """Read COCO annotations, prediction-style lists, or image_id -> captions mappings."""
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(obj, dict) and 'annotations' in obj:
        obj = obj['annotations']
    out = defaultdict(list)
    if isinstance(obj, dict):
        for image_id, captions in obj.items():
            if isinstance(captions, str):
                captions = [captions]
            if not isinstance(captions, list):
                raise ValueError(f'Invalid captions for image_id {image_id}')
            for caption in captions:
                if isinstance(caption, dict):
                    caption = caption.get('caption', '')
                if isinstance(caption, str) and caption.strip():
                    out[str(image_id)].append(caption.strip())
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and 'image_id' in item and 'caption' in item:
                out[str(item['image_id'])].append(str(item['caption']).strip())
    else:
        raise ValueError('References must be COCO annotations or image_id-to-captions mapping')
    if not out:
        raise ValueError(f'No reference captions parsed from {path}')
    return dict(out)


def f1_unigram(candidate, reference):
    a, b = Counter(tokenize(candidate)), Counter(tokenize(reference))
    match = sum((a & b).values())
    if not a or not b or not match:
        return 0.0
    p, r = match / sum(a.values()), match / sum(b.values())
    return 2*p*r/(p+r)

def ref_score(candidate, references):
    # Diagnostic best-reference unigram F1; NOT CIDEr, SPICE or official BLEU.
    return max((f1_unigram(candidate, r) for r in references), default=None)

def categories_for(refs, baseline, finetuned):
    # Prefer ground-truth references; if absent categories are prediction-derived.
    source = ' '.join(refs) if refs else baseline + ' ' + finetuned
    words = set(tokenize(source))
    matches = [name for name, vocab in CATEGORIES.items() if words & vocab]
    return matches or ['other']

def bootstrap_ci(deltas, n=2000, seed=42):
    if not deltas:
        return None
    rng = random.Random(seed)
    size = len(deltas)
    means = [sum(deltas[rng.randrange(size)] for _ in range(size))/size for _ in range(n)]
    means.sort()
    return [means[int(0.025*(n-1))], means[int(0.975*(n-1))]]

def write_csv(path, rows, columns):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', required=True, help='Baseline prediction JSON')
    ap.add_argument('--finetuned', required=True, help='Fine-tuned prediction JSON')
    ap.add_argument('--references', help='Optional COCO references JSON with annotations')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--top_k', type=int, default=100)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.top_k < 1 or args.bootstrap < 0:
        ap.error('--top_k must be >= 1 and --bootstrap must be >= 0')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = load_preds(args.baseline)
    ft = load_preds(args.finetuned)
    refs = load_refs(args.references)
    if args.references:
        common_ref_ids = set(refs) & set(base) & set(ft)
        print(f'References: {len(refs)} image IDs; matched: {len(common_ref_ids)}', flush=True)
        if not common_ref_ids:
            raise ValueError('No reference image IDs match predictions; check ID format and test split')
    ids = sorted(set(base) & set(ft))
    if not ids:
        raise SystemExit('No common image_ids found; check prediction files')
    print(f'Baseline: {len(base)} | Fine-tuned: {len(ft)} | Common: {len(ids)} | References: {len(refs)}')
    rows = []
    groups = defaultdict(list)
    deltas = []
    changed = 0
    for image_id in ids:
        b, f = base[image_id], ft[image_id]
        r = refs.get(image_id, [])
        is_changed = tokenize(b) != tokenize(f)
        changed += is_changed
        bs, fs = ref_score(b, r), ref_score(f, r)
        delta = (fs - bs) if bs is not None else None
        if delta is not None:
            deltas.append(delta)
        cats = categories_for(r, b, f)
        row = {
            'image_id': image_id, 'baseline_caption': b, 'finetuned_caption': f,
            'changed': int(is_changed), 'baseline_words': len(tokenize(b)),
            'finetuned_words': len(tokenize(f)), 'baseline_ref_f1': bs,
            'finetuned_ref_f1': fs, 'delta_ref_f1': delta,
            'categories': '|'.join(cats), 'references': ' || '.join(r),
        }
        rows.append(row)
        for cat in cats:
            groups[cat].append(row)
    columns = list(rows[0])
    write_csv(output / 'paired_captions.csv', rows, columns)
    category_summary = []
    for name, group in sorted(groups.items()):
        scored = [x for x in group if x['delta_ref_f1'] is not None]
        category_summary.append({
            'category': name, 'n_images': len(group),
            'changed_fraction': sum(x['changed'] for x in group)/len(group),
            'mean_delta_ref_f1': (sum(x['delta_ref_f1'] for x in scored)/len(scored)) if scored else None,
            'n_with_references': len(scored),
        })
    write_csv(output / 'category_summary.csv', category_summary,
              ['category', 'n_images', 'changed_fraction', 'mean_delta_ref_f1', 'n_with_references'])
    scored_rows = [x for x in rows if x['delta_ref_f1'] is not None and x['changed']]
    write_csv(output / 'top_improvements.csv',
              sorted(scored_rows, key=lambda x: x['delta_ref_f1'], reverse=True)[:args.top_k], columns)
    write_csv(output / 'top_regressions.csv',
              sorted(scored_rows, key=lambda x: x['delta_ref_f1'])[:args.top_k], columns)
    summary = {
        'baseline_file': args.baseline, 'finetuned_file': args.finetuned,
        'references_file': args.references, 'baseline_n': len(base), 'finetuned_n': len(ft),
        'common_images': len(ids), 'baseline_only': len(base)-len(ids),
        'finetuned_only': len(ft)-len(ids), 'changed_images': changed,
        'changed_fraction': changed/len(ids),
        'baseline_mean_words': sum(x['baseline_words'] for x in rows)/len(rows),
        'finetuned_mean_words': sum(x['finetuned_words'] for x in rows)/len(rows),
        'images_with_references': len(deltas),
        'mean_delta_best_reference_unigram_f1': sum(deltas)/len(deltas) if deltas else None,
        'bootstrap_95pct_ci_delta': bootstrap_ci(deltas, args.bootstrap, args.seed) if args.bootstrap else None,
        'num_improved': sum(d > 1e-12 for d in deltas),
        'num_regressed': sum(d < -1e-12 for d in deltas),
        'num_tied': sum(abs(d) <= 1e-12 for d in deltas),
        'notes': [
            'Best-reference unigram F1 is a lightweight diagnostic, not official COCO BLEU/CIDEr/SPICE.',
            'Categories use simple English keyword rules, may overlap, and are not a validated semantic classifier.',
            'If references are omitted, only caption changes and heuristic categories can be analyzed.',
            'If references are provided, they should come from the matching evaluation split and image IDs.',
            'Confidence interval bootstraps image-level paired differences, conditional on this evaluation set.',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Files written to {output}')

if __name__ == '__main__':
    main()
