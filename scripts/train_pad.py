#!/usr/bin/env python3
"""Train the PAD-UFES-20 skin-lesion classifier (sgd / dpsgd / feature_dpsgd).

Modes:
  sgd            standard (non-private) Adam training
  dpsgd          DP-SGD on the full (image+meta) gradient, full-batch
  feature_dpsgd  Feature DP-SGD: image vs metadata gradient components are
                 clipped separately (C_base / C_meta); noise scales with the
                 metadata clip only

Image features (EfficientNet-B3, frozen) are extracted once and cached
(--cache_path) so repeated runs (e.g. multiple seeds) don't redo feature
extraction. The train/val/test split is patient-level, approximately
label-stratified, and fixed by --split_seed (independent of --seed), so all
runs in a sweep see the same split; --seed controls model init / shuffling /
DP noise.

Also logs subgroup test accuracies (age bins, gender, family background,
smoke, drink, Fitzpatrick, cancer history) — printed to stdout AND appended
as rows to a CSV (--subgroup_csv).

Prints a single machine-parsable result line at the end:
  RESULT mode=<mode> lr=<lr> epsilon=<eps_or_NA> seed=<seed> test_loss=<..> test_acc=<..>

Example (see docs/pad_ufes.md for the full walkthrough):

    python -m scripts.train_pad --mode sgd --lr 1e-3 --use_metadata \
        --data_root /data/PAD-UFES-20 --download
    python -m scripts.train_pad --mode feature_dpsgd --lr 1e-3 --epsilon 8 \
        --data_root /data/PAD-UFES-20
"""

import argparse
import os

from scripts._bootstrap import setup_project_paths

setup_project_paths()


def main():
    p = argparse.ArgumentParser(
        description="Train PAD-UFES-20 MLP under sgd / dpsgd / feature_dpsgd"
    )
    p.add_argument("--mode", required=True, choices=["sgd", "dpsgd", "feature_dpsgd"])
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epsilon", type=float, default=None,
                   help="Target DP epsilon. Required for dpsgd / feature_dpsgd, ignored for sgd.")
    p.add_argument("--delta", type=float, default=None,
                   help="Target DP delta. Default: 1/N_train**1.1 (as in the original notebook).")
    p.add_argument("--seed", type=int, default=0, help="Run seed: model init / shuffling / DP noise.")
    p.add_argument("--split_seed", type=int, default=101,
                   help="Fixed seed for the patient-level, approximately label-stratified split.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="C for --mode dpsgd.")
    p.add_argument("--C_base", type=float, default=50.0, help="Image-feature clip norm for feature_dpsgd.")
    p.add_argument("--C_meta", type=float, default=1.0, help="Meta-feature clip norm for feature_dpsgd.")
    p.add_argument("--use_metadata", action="store_true",
                   help="Concatenate real metadata one-hot features. Only meaningful for "
                        "--mode sgd: omit for the 'no meta' setting (EfficientNet features + "
                        "a zero vector of the usual metadata size). For --mode dpsgd / "
                        "feature_dpsgd this is always forced True — feature_dpsgd's base/meta "
                        "gradient split (zeroing metadata columns to isolate the image-only "
                        "gradient) is meaningless if the metadata block is already zero. "
                        "Switching this flag never re-triggers feature extraction — only the "
                        "cheap raw-feature combine step re-runs.")
    p.add_argument("--data_root", type=str, required=True,
                   help="PAD-UFES-20 root directory (metadata.csv + all_images/; "
                        "created by --download if missing).")
    p.add_argument("--img_dir", type=str, default=None,
                   help="Default: <data_root>/all_images")
    p.add_argument("--meta_path", type=str, default=None,
                   help="Default: <data_root>/metadata.csv")
    p.add_argument("--download", action="store_true", help="Download+extract dataset if img_dir is empty.")
    p.add_argument("--cache_path", type=str, default=None,
                   help="Where extracted image features are cached (shared across all runs/seeds). "
                        "Default: <data_root>/features_cache_patient_ohe.npz")
    p.add_argument("--subgroup_csv", type=str, default="subgroup_results.csv",
                   help="CSV that subgroup accuracy rows are appended to (long format, one row "
                        "per subgroup value per run).")
    p.add_argument("--setting_tag", type=str, default=None,
                   help="Optional short label for this run (e.g. 'dpsgd_eps4') stored in the "
                        "subgroup CSV; defaults to '<mode>_eps<epsilon>' if not given.")
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value, e.g. '0'.")
    args = p.parse_args()

    if args.mode in ("dpsgd", "feature_dpsgd") and args.epsilon is None:
        p.error(f"--epsilon is required for --mode {args.mode}")

    if args.mode in ("dpsgd", "feature_dpsgd") and not args.use_metadata:
        print(f"Note: --mode {args.mode} always trains on real metadata "
              f"(required for the base/meta gradient split); forcing use_metadata=True.")
        args.use_metadata = True

    if args.img_dir is None:
        args.img_dir = os.path.join(args.data_root, "all_images")
    if args.meta_path is None:
        args.meta_path = os.path.join(args.data_root, "metadata.csv")
    if args.cache_path is None:
        args.cache_path = os.path.join(args.data_root, "features_cache_patient_ohe.npz")

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Torch imports happen after CUDA_VISIBLE_DEVICES is set.
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from medmap.models.pad_mlp import MLP1Layer, NUM_CLASSES
    from medmap.pad.evaluation import compute_subgroup_accuracies, compute_test_metrics
    from medmap.pad.features import CombinedDataset, combine_features, load_or_build_raw_features
    from medmap.pad.preprocessing import (
        patient_stratified_split,
        print_split_class_distributions,
        set_seed,
    )
    from medmap.pad.privacy import (
        find_noise_multiplier,
        run_training_dpsgd,
        run_training_feature_dpsgd,
    )
    from medmap.pad.training import evaluate, run_training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Data / features: built once, cached, and shared across seeds and across
    #    --use_metadata settings (the combine step below is cheap and re-runs
    #    every time, but the expensive EfficientNet extraction only runs once).
    img_features, meta_ohe, y_all, img_dim, class_names, subgroups, patient_ids, meta_feature_names = (
        load_or_build_raw_features(
            args.cache_path, args.meta_path, args.img_dir, args.data_root,
            args.download, device,
        )
    )
    x_combined = combine_features(img_features, meta_ohe, args.use_metadata)
    print(f"use_metadata={args.use_metadata}  x_combined shape={x_combined.shape}")

    # 2) Fixed patient-level, approximately label-stratified split.
    idx_tr, idx_va, idx_te = patient_stratified_split(
        patient_ids, y_all, args.split_seed
    )

    x_tr, y_tr = x_combined[idx_tr], y_all[idx_tr]
    x_va, y_va = x_combined[idx_va], y_all[idx_va]
    x_te, y_te = x_combined[idx_te], y_all[idx_te]
    subgroup_te = {col: arr.iloc[idx_te].to_numpy() if isinstance(arr, pd.Series)
                   else np.asarray(arr)[idx_te] for col, arr in subgroups.items()}

    print(f"Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}")
    print_split_class_distributions(
        {'Train': idx_tr, 'Validation': idx_va, 'Test': idx_te},
        y_all, patient_ids, class_names
    )
    input_dim = x_tr.shape[1]

    # 3) Seed the run (model init / shuffling / DP noise) *after* the fixed split.
    set_seed(args.seed)

    # 4) Loaders — batch size depends on mode.
    train_batch = len(y_tr)
    train_loader = DataLoader(CombinedDataset(x_tr, y_tr), batch_size=train_batch,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(CombinedDataset(x_va, y_va), batch_size=max(len(y_va), 1),
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(CombinedDataset(x_te, y_te), batch_size=max(len(y_te), 1),
                              shuffle=False, num_workers=2)

    # 5) Model.
    model = MLP1Layer(input_dim, hidden_dim=args.hidden_dim, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()

    delta = args.delta if args.delta is not None else 1 / (len(y_tr) ** 1.1)

    # 6) Train.
    label = f"{args.mode} lr={args.lr} seed={args.seed}"
    if args.mode == "sgd":
        run_training(model, train_loader, val_loader, device,
                     epochs=args.epochs, lr=args.lr, label=label)
    elif args.mode == "dpsgd":
        noise_mult = find_noise_multiplier(1.0, args.epochs, args.epsilon, delta)
        print(f"DP-SGD: epsilon={args.epsilon} delta={delta:.3e} -> noise_multiplier={noise_mult:.4f}")
        run_training_dpsgd(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr,
                            max_grad_norm=args.max_grad_norm, noise_multiplier=noise_mult, label=label)
    else:  # feature_dpsgd
        noise_mult = find_noise_multiplier(1.0, args.epochs, args.epsilon, delta)
        print(f"Feature DP-SGD: epsilon={args.epsilon} delta={delta:.3e} -> noise_multiplier={noise_mult:.4f}")
        run_training_feature_dpsgd(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr,
                                    C_base=args.C_base, C_meta=args.C_meta, noise_multiplier=noise_mult,
                                    img_dim=img_dim, label=label)

    # 7) Test.
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    eps_str = args.epsilon if args.epsilon is not None else "NA"
    print(f"[{label}] TEST loss={test_loss:.4f} acc={test_acc:.4f}")
    print(f"RESULT mode={args.mode} lr={args.lr} epsilon={eps_str} seed={args.seed} "
          f"use_metadata={args.use_metadata} "
          f"test_loss={test_loss:.4f} test_acc={round(test_acc * 100, 2)}")

    # 8) Overall/per-class metrics + subgroup accuracy breakdown.
    compute_test_metrics(model, x_te, y_te, device, class_names)

    setting_tag = args.setting_tag or f"{args.mode}_eps{eps_str}"
    run_meta = {
        "setting": setting_tag,
        "mode": args.mode,
        "lr": args.lr,
        "epsilon": eps_str,
        "use_metadata": args.use_metadata,
        "seed": args.seed,
    }
    compute_subgroup_accuracies(
        model, x_te, y_te, subgroup_te, device,
        run_meta, args.subgroup_csv
    )


if __name__ == "__main__":
    main()
