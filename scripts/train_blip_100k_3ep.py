#!/usr/bin/env python3
"""Single-GPU BLIP COCO caption fine-tuning: 100k annotations, 3 epochs with per-epoch validation.

Run from the root of the original BLIP repository. Does not modify train_caption.py.
"""
import argparse
import json
import shutil
import random
import time
from pathlib import Path

import numpy as np
import torch
from ruamel import yaml

from data import create_dataset, create_loader
from data.utils import coco_caption_eval
from models.blip import blip_decoder
from train_caption import evaluate


def log(message):
    print(message, flush=True)


def save_single_gpu_result(results, result_dir, filename):
    """Save caption predictions without torch.distributed barriers."""
    path = Path(result_dir) / f'{filename}.json'
    dedup = {}
    for row in results:
        dedup[int(row['image_id'])] = {'image_id': int(row['image_id']), 'caption': row['caption']}
    with path.open('w', encoding='utf-8') as f:
        json.dump(list(dedup.values()), f, ensure_ascii=False)
    log(f'Saved {len(dedup)} captions to {path}')
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='output/caption_coco_smoke_10k/config.yaml')
    parser.add_argument('--pretrained', default='/home/msai/li0034qi/blip_project/checkpoints/pretrained/model_base_caption_capfilt_large.pth')
    parser.add_argument('--output_dir', default='output/caption_coco_100k_3ep_lr5e6')
    parser.add_argument('--train_samples', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error('--epochs must be >= 1')
    if args.train_samples <= 0:
        parser.error('--train_samples must be positive')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_dir = out / 'result'
    result_dir.mkdir(exist_ok=True)
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config['pretrained'] = args.pretrained  # avoid redownloading from the URL
    config['init_lr'] = args.lr
    config['max_epoch'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(f'Pretrained weights missing: {args.pretrained}')
    with open(out / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: submit this script via sbatch on a GPU compute node')
    device = torch.device('cuda')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    total_start = time.monotonic()
    log(f'GPU: {torch.cuda.get_device_name(0)}')
    log(f'Config: {args.config} | batch_size={config["batch_size"]} | lr={args.lr} | epochs={args.epochs}')
    log('Loading COCO datasets ...')
    train_ds, val_ds, test_ds = create_dataset('caption_coco', config)
    original_size = len(train_ds)
    if not hasattr(train_ds, 'annotation'):
        raise RuntimeError('Expected COCO training dataset to expose .annotation')
    if args.train_samples > original_size:
        raise ValueError(f'Requested {args.train_samples} annotations, only {original_size} available')
    # Same first-N slicing as the previous smoke test; preserves comparability.
    train_ds.annotation = train_ds.annotation[:args.train_samples]
    log(f'Train annotations: {len(train_ds)} / {original_size}; val images: {len(val_ds)}; test images: {len(test_ds)}')
    unique_train_images = len({str(a.get('image', '')) for a in train_ds.annotation})
    log(f'Unique training image paths in selected annotations: {unique_train_images}')
    loaders = create_loader(
        [train_ds, val_ds, test_ds], [None, None, None],
        batch_size=[config['batch_size']] * 3,
        num_workers=[args.num_workers] * 3,
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )
    train_loader, val_loader, test_loader = loaders
    log(f'Training batches: {len(train_loader)}')

    log('Loading original BLIP checkpoint ...')
    model = blip_decoder(
        pretrained=config['pretrained'], image_size=config['image_size'],
        vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config['weight_decay'])
    # Fixed learning rate to isolate the effect of additional epochs.
    # Every epoch uses 5e-6; no cosine decay or warmup is applied.
    torch.cuda.reset_peak_memory_stats()
    best_cider = float('-inf')
    best_epoch = None
    best_path = out / 'checkpoint_best.pth'
    history = []
    training_seconds = 0.0
    validation_seconds = 0.0
    test_seconds = 0.0

    def write_summary(test_metrics=None, test_json=None):
        summary = {
            'train_annotations': len(train_ds),
            'unique_train_image_paths': unique_train_images,
            'original_train_annotations': original_size,
            'epochs_requested': args.epochs,
            'epochs_completed': len(history),
            'batch_size': config['batch_size'],
            'learning_rate': args.lr,
            'learning_rate_schedule': 'constant',
            'history': history,
            'best_epoch': best_epoch,
            'best_val_CIDEr': best_cider if best_epoch is not None else None,
            'checkpoint': str(best_path) if best_epoch is not None else None,
            'training_minutes': training_seconds / 60,
            'validation_minutes': validation_seconds / 60,
            'test_minutes': test_seconds / 60,
            'total_minutes': (time.monotonic() - total_start) / 60,
            'peak_vram_allocated_gb': torch.cuda.max_memory_allocated() / 2**30,
            'peak_vram_reserved_gb': torch.cuda.max_memory_reserved() / 2**30,
            'test': test_metrics,
            'test_results': test_json,
        }
        tmp = out / 'experiment_summary.tmp.json'
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        tmp.replace(out / 'experiment_summary.json')
        return summary

    for epoch in range(args.epochs):
        log(f'===== EPOCH {epoch + 1}/{args.epochs}: TRAIN START =====')
        model.train()
        epoch_start = time.monotonic()
        total_loss = 0.0
        for step, (images, captions, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model(images, captions)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                elapsed = time.monotonic() - epoch_start
                rate = step / max(elapsed, 1.0)
                eta = (len(train_loader) - step) / max(rate, 1e-9) / 60
                log(f'EPOCH {epoch + 1} step={step}/{len(train_loader)} '
                    f'loss={loss.item():.4f} avg_loss={total_loss/step:.4f} '
                    f'elapsed_min={elapsed/60:.1f} eta_min={eta:.1f} '
                    f'peak_vram_gb={torch.cuda.max_memory_allocated()/2**30:.2f}')
        torch.cuda.synchronize()
        epoch_train_seconds = time.monotonic() - epoch_start
        training_seconds += epoch_train_seconds
        avg_loss = total_loss / len(train_loader)

        # Save a recovery checkpoint BEFORE running validation. It is overwritten
        # every epoch, keeping disk usage low and preserving weights on eval failure.
        recovery_path = out / 'checkpoint_last.pth'
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': dict(config)}, recovery_path)
        log(f'Saved recovery checkpoint: {recovery_path}')

        log(f'===== EPOCH {epoch + 1}/{args.epochs}: VALIDATION START =====')
        val_start = time.monotonic()
        val_json = save_single_gpu_result(
            evaluate(model, val_loader, device, config), result_dir, f'val_epoch{epoch}')
        val_metrics = {k: float(v) for k, v in
                       coco_caption_eval(config['coco_gt_root'], val_json, 'val').eval.items()}
        epoch_val_seconds = time.monotonic() - val_start
        validation_seconds += epoch_val_seconds
        cider = val_metrics['CIDEr']
        log(f'EPOCH {epoch + 1} loss={avg_loss:.5f} val_CIDEr={cider:.6f} '
            f'val_Bleu_4={val_metrics.get("Bleu_4", float("nan")):.6f}')

        if cider > best_cider:
            best_cider = cider
            best_epoch = epoch + 1
            # Copy checkpoint_last to checkpoint_best without re-serializing model.
            shutil.copy2(recovery_path, best_path)
            log(f'NEW BEST: epoch={best_epoch} CIDEr={best_cider:.6f}; saved {best_path}')

        history.append({
            'epoch': epoch + 1,
            'train_mean_loss': avg_loss,
            'train_minutes': epoch_train_seconds / 60,
            'val_minutes': epoch_val_seconds / 60,
            'val': val_metrics,
            'val_results': val_json,
        })
        write_summary()

    # Test only once, with the BEST validation checkpoint (not necessarily epoch 3).
    log(f'===== TEST BEST CHECKPOINT: EPOCH {best_epoch} =====')
    try:
        checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(best_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    test_start = time.monotonic()
    test_json = save_single_gpu_result(
        evaluate(model, test_loader, device, config), result_dir, 'test_best')
    test_metrics = {k: float(v) for k, v in
                    coco_caption_eval(config['coco_gt_root'], test_json, 'test').eval.items()}
    test_seconds = time.monotonic() - test_start
    summary = write_summary(test_metrics, test_json)
    log('===== FINISHED =====')
    log(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
