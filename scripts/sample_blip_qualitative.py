#!/usr/bin/env python3
"""Sample 10 improvements + 10 regressions and create caption comparison contact sheets."""
import argparse
import csv
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def image_index(annotation_path):
    data = json.loads(Path(annotation_path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('images', data.get('annotations', []))
    result = {}
    for row in data:
        if isinstance(row, dict) and 'image_id' in row:
            result[str(row['image_id'])] = row.get('image', row.get('file_name', ''))
        elif isinstance(row, dict) and 'id' in row:
            result[str(row['id'])] = row.get('file_name', '')
    return result


def locate(image_id, relative, roots):
    candidates = []
    if relative:
        p = Path(relative)
        candidates.append(p)
        for root in roots:
            candidates.extend([root / p, root / p.name])
    try:
        num = int(image_id)
    except ValueError:
        return None
    for root in roots:
        for split in ('val2014', 'train2014', 'test2014', ''):
            for prefix in ('COCO_val2014_', 'COCO_train2014_', 'COCO_test2014_'):
                candidates.append(root / split / f'{prefix}{num:012d}.jpg')
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def choose(rows, positive, n, pool, seed):
    eligible = [r for r in rows if r.get('changed', '1') in ('1', 'True', 'true') and
                ((float(r['delta_ref_f1']) > 0) if positive else (float(r['delta_ref_f1']) < 0))]
    eligible.sort(key=lambda r: float(r['delta_ref_f1']), reverse=positive)
    shortlist = eligible[:pool] if pool > 0 else eligible
    if len(shortlist) < n:
        raise ValueError(f'Only {len(shortlist)} eligible samples; need {n}')
    return random.Random(seed).sample(shortlist, n)


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, current = [], ''
    for word in words:
        trial = (current + ' ' + word).strip()
        if draw.textbbox((0, 0), trial, font=font)[2] > max_width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def make_sheet(items, label, out_path, thumb_w=480):
    font = ImageFont.load_default()
    width = 1040
    margin = 20
    card_heights = []
    for item in items:
        refs = item['references'].split(' || ')
        text = [f"Image {item['image_id']}   delta F1: {float(item['delta_ref_f1']):+.4f}",
                'Baseline: ' + item['baseline_caption'],
                'Fine-tuned: ' + item['finetuned_caption']]
        text += [f'Ref {i+1}: {r}' for i, r in enumerate(refs)]
        dummy = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        lines = sum((len(wrap_text(dummy, t, font, width-thumb_w-4*margin)) for t in text), 0)
        card_heights.append(max(310, 40 + lines*16))
    canvas = Image.new('RGB', (width, 70 + sum(card_heights) + margin*(len(items)+1)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), f'{label} - 10 sampled images', fill='black', font=font)
    y = 70
    for item, h in zip(items, card_heights):
        draw.rectangle((margin, y, width-margin, y+h), outline='#bbbbbb', width=2)
        if item['image_path']:
            with Image.open(item['image_path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((thumb_w, h-20))
                canvas.paste(im, (margin+10+(thumb_w-im.width)//2, y+10+(h-20-im.height)//2))
        else:
            draw.text((margin+20, y+100), 'IMAGE NOT FOUND', fill='#cc0000', font=font)
        x = thumb_w+2*margin
        yy = y+12
        fields = [f"Image {item['image_id']}  delta F1: {float(item['delta_ref_f1']):+.4f}",
                  'Baseline: '+item['baseline_caption'], 'Fine-tuned: '+item['finetuned_caption']]
        fields += [f'Ref {i+1}: {r}' for i,r in enumerate(item['references'].split(' || '))]
        for field in fields:
            for line in wrap_text(draw, field, font, width-x-2*margin):
                draw.text((x, yy), line, fill='black', font=font)
                yy += 16
            yy += 5
        y += h+margin
    canvas.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis_dir', default='output/caption_analysis_lr1e6_with_refs')
    p.add_argument('--annotation', default='annotation/coco_karpathy_test.json')
    p.add_argument('--image_root', action='append', required=True,
                   help='COCO image directory; repeat flag for multiple roots')
    p.add_argument('--output_dir', default='output/blip_qualitative_10x2')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--pool', type=int, default=100,
                   help='Random sample from top N positive/negative F1 changes; 0=all')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.n < 1 or args.pool < 0 or (args.pool and args.pool < args.n):
        p.error('Require n>=1 and pool=0 or pool>=n')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(s).expanduser() for s in args.image_root]
    index = image_index(args.annotation)
    groups = [('improvements', 'top_improvements.csv', True),
              ('regressions', 'top_regressions.csv', False)]
    all_rows = []
    for group, filename, positive in groups:
        rows = read_rows(Path(args.analysis_dir)/filename)
        selected = choose(rows, positive, args.n, args.pool, args.seed)
        for r in selected:
            path = locate(r['image_id'], index.get(r['image_id'], ''), roots)
            r['image_path'] = str(path) if path else ''
            r['group'] = group
            all_rows.append(r)
        make_sheet(selected, group, out/f'{group}_contact_sheet.jpg')
        print(f'{group}: {len(selected)} selected, {sum(bool(r["image_path"]) for r in selected)} images found')
    cols = ['group','image_id','image_path','delta_ref_f1','baseline_ref_f1',
            'finetuned_ref_f1','baseline_caption','finetuned_caption','references','categories']
    with open(out/'selected_20.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    missing = [r['image_id'] for r in all_rows if not r['image_path']]
    (out/'missing_images.txt').write_text('\n'.join(missing), encoding='utf-8')
    print('Saved to:', out)
    if missing:
        print('WARNING: Missing images for IDs:', ', '.join(missing))
        print('Pass the correct --image_root (e.g. COCO images directory).')

if __name__ == '__main__':
    main()
