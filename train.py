"""
FairUnGSL Training Script.

Usage:
    python train.py --dataset german --epochs 200 --device cuda:0
    python train.py --dataset bail --epochs 300 --seed 42
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

# Expose src/ on sys.path so that core modules can be imported by name from
# both this file and any sub-module under models/.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fairness_utils import load_dataset, sparse_mx_to_torch_sparse_tensor, compute_fairness_metrics, feature_norm
from fairness_loss import FairnessLoss
from adversarial import SensitiveDiscriminator, compute_adv_loss
from fair_contrastive import FairContrastiveHead
from gcn import GCN, normalize_adj, normalize_adj_sparse
# normalize_adj imported for contrastive loss path


# ============================================================
# Entropy Computation (Pre-training step)
# ============================================================

def compute_entropy_standalone(features, adj, labels, train_mask, val_mask,
                                n_feat, n_class, device, n_runs=3,
                                n_epochs=200, lr=0.01, hidden=64, dropout=0.5,
                                class_weights=None):
    """
    Standalone entropy computation.

    Parameters
    ----------
    features, adj, labels : torch.Tensor
        Graph data on device.
    train_mask, val_mask : torch.Tensor
        Boolean masks.
    n_feat, n_class : int
        Feature and class dimensions.
    device : str
        Device.
    n_runs : int
        Number of GCN training runs.
    n_epochs : int
        Epochs per run.
    lr : float
        Learning rate.
    hidden : int
        Hidden dimension.
    dropout : float
        Dropout rate.

    Returns
    -------
    torch.Tensor
        Averaged entropy vector [N] on device.
    """
    # Use sparse normalization for large graphs to avoid OOM
    n_nodes = adj.shape[0]
    if n_nodes > 5000:
        adj_sp = adj.to_sparse() if not adj.is_sparse else adj
        adj_norm = normalize_adj_sparse(adj_sp)
    else:
        adj_norm = normalize_adj(adj)
    all_entropies = []

    for run in range(n_runs):
        seed = 42 + run * 100
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GCN(n_feat, hidden, n_class, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(features, adj_norm)
            loss = F.cross_entropy(output[train_mask], labels[train_mask], weight=class_weights)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(features, adj_norm)
                val_loss = F.cross_entropy(val_output[val_mask], labels[val_mask], weight=class_weights)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())

        # Load best model and compute entropy
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            output = model(features, adj_norm)
            probs = F.softmax(output, dim=1)
            # Shannon entropy: H = -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)  # [N]

        all_entropies.append(entropy)
        print(f"  Entropy run {run + 1}/{n_runs}: mean={entropy.mean():.4f}, max={entropy.max():.4f}")

    # Average entropies across runs
    avg_entropy = torch.stack(all_entropies).mean(dim=0)
    return avg_entropy


# ============================================================
# Model Selection: Algorithm 1 (Paper 16, KDD 2024)
# ============================================================

def algorithm1_select(epoch_history):
    """Select best epoch using Algorithm 1 from FairGraphBench (KDD 2024).

    Finds the epoch with best fairness (min SP+EO) while maintaining
    ACC, AUC, F1 above a threshold ratio of their respective maxima.
    Scans ratios from 0.95 down to 0.90.

    Parameters
    ----------
    epoch_history : list of dict
        Each dict has keys: epoch, acc, auc, f1, sp, eo, state_dict.

    Returns
    -------
    dict or None
        The selected entry from epoch_history, or None if empty.
    """
    if not epoch_history:
        return None

    max_acc = max(e['acc'] for e in epoch_history)
    max_auc = max(e['auc'] for e in epoch_history)
    max_f1  = max(e['f1']  for e in epoch_history)

    best_fairness = float('inf')
    best_entry = None
    threshold_ratios = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]

    for ratio in threshold_ratios:
        thr_acc = max_acc * ratio
        thr_auc = max_auc * ratio
        thr_f1  = max_f1  * ratio
        for entry in epoch_history:
            if (entry['acc'] >= thr_acc and
                entry['auc'] >= thr_auc and
                entry['f1']  >= thr_f1):
                fairness = entry['sp'] + entry['eo']
                if fairness < best_fairness:
                    best_fairness = fairness
                    best_entry = entry

    if best_entry is None:
        best_entry = min(epoch_history, key=lambda e: e['sp'] + e['eo'])

    return best_entry


# ============================================================
# Training Loop
# ============================================================

def train_fair_grcn(conf, device='cuda:0'):
    """
    Main training function for FairGRCN + FairUnGSL.

    Parameters
    ----------
    conf : dict
        Configuration dictionary.
    device : str
        Device to use.
    """
    dataset_name = conf['dataset']
    data_root = conf.get('data_root', 'dataset/')

    print(f"=" * 60)
    print(f"FairUnGSL Training: {dataset_name}")
    print(f"=" * 60)

    # --- Load dataset ---
    print(f"\n[1/5] Loading dataset: {dataset_name}")
    adj_sp, features, labels, idx_train, idx_val, idx_test, sensitive = \
        load_dataset(dataset_name, data_root)

    # === minority label ratio patch ===
    _min_ratio = conf.get('minority_label_ratio', 1.0)
    if _min_ratio < 0.9999:
        from measure_utils import apply_minority_label_ratio
        _seed_for_ratio = conf.get('_run_seed', conf.get('seed', 42))
        idx_train, _ratio_info = apply_minority_label_ratio(idx_train, sensitive, labels, _min_ratio, _seed_for_ratio)
        print(f"  [minority_label_ratio={_min_ratio}] train_g0={_ratio_info['n_train_g0']}, train_g1={_ratio_info['n_train_g1_kept']} (dropped {_ratio_info['n_train_g1_dropped']})")
    # === end patch ===

    n_nodes = features.shape[0]
    n_feat = features.shape[1]
    n_class = labels.max().item() + 1

    print(f"  Nodes: {n_nodes}, Features: {n_feat}, Classes: {n_class}")
    print(f"  Sensitive groups: 0={int((sensitive == 0).sum())}, 1={int((sensitive == 1).sum())}")
    print(f"  Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    # Check label distribution
    for c in range(labels.max().item() + 1):
        n_c = (labels == c).sum().item()
        n_train_c = (labels[idx_train] == c).sum().item()
        print(f"  Class {c}: total={n_c}, train={n_train_c}")

    # Normalize features and convert to torch tensors
    features = feature_norm(features)
    adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)
    features = features.to(device)
    labels = labels.to(device)
    sensitive = sensitive.to(device)

    # Input feature debiasing (v5: remove sensitive direction from features)
    debias_alpha = conf.get('debias_alpha', 0.0)  # 0 = disabled
    if debias_alpha > 0:
        with torch.no_grad():
            feat_0 = features[sensitive == 0].mean(dim=0)
            feat_1 = features[sensitive == 1].mean(dim=0)
            direction = feat_1 - feat_0
            direction = direction / direction.norm().clamp(min=1e-8)
            projection = (features @ direction).unsqueeze(1) * direction.unsqueeze(0)
            features = features - debias_alpha * projection
        print(f"  Feature debiasing applied (alpha={debias_alpha})")

    # v7: Counterfactual feature generation
    # Two modes:
    #   cf_mode='simple' (default): group-mean shift (delta = mean_1 - mean_0)
    #   cf_mode='label_cond': label-conditional counterfactual
    #     (delta per label y = mean(X|y,s=1) - mean(X|y,s=0))
    #     Preserves label semantics while flipping sensitive attribute
    cf_weight = conf.get('cf_weight', 0.0)  # 0 = disabled
    cf_mode = conf.get('cf_mode', 'simple')
    cf_shift_scale = conf.get('cf_shift_scale', 1.0)  # 1.0 = full flip, <1 = partial
    cf_features = None
    if cf_weight > 0:
        with torch.no_grad():
            s = sensitive.float().to(device)
            shift_sign = 1.0 - 2.0 * s.unsqueeze(1)  # [N,1]: +1 if s=0, -1 if s=1

            if cf_mode == 'label_cond':
                # Label-conditional: shift within same-label, opposite-group direction
                cf_features = features.clone()
                for y in torch.unique(labels):
                    y_mask = (labels == y)
                    g0_mask = y_mask & (sensitive == 0)
                    g1_mask = y_mask & (sensitive == 1)
                    if g0_mask.sum() == 0 or g1_mask.sum() == 0:
                        continue
                    mean_g0 = features[g0_mask].mean(dim=0)
                    mean_g1 = features[g1_mask].mean(dim=0)
                    delta_y = (mean_g1 - mean_g0) * cf_shift_scale
                    # Apply shift only to nodes with this label
                    affected = y_mask.to(device)
                    shift_per_node = delta_y.unsqueeze(0) * shift_sign  # [N,F]
                    cf_features[affected] = features[affected] + shift_per_node[affected]
                print(f"  Counterfactual features generated (label_cond, weight={cf_weight}, scale={cf_shift_scale})")
            else:
                # Simple group-mean shift
                feat_0_mean = features[sensitive == 0].mean(dim=0)
                feat_1_mean = features[sensitive == 1].mean(dim=0)
                group_delta = (feat_1_mean - feat_0_mean) * cf_shift_scale
                cf_features = features + group_delta.unsqueeze(0) * shift_sign
                print(f"  Counterfactual features generated (simple, weight={cf_weight}, scale={cf_shift_scale})")

    # Create boolean masks
    train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # --- Compute or load entropy ---
    backbone_prefix = conf.get('backbone', 'GRCN').lower()
    entropy_path = os.path.join(
        conf.get('entropy_dir', 'entropy/'),
        f"{backbone_prefix}_{dataset_name}_entropy.pt"
    )
    os.makedirs(os.path.dirname(entropy_path), exist_ok=True)

    if os.path.exists(entropy_path):
        print(f"\n[2/5] Loading pre-computed entropy from {entropy_path}")
        entropy = torch.load(entropy_path, map_location=device)
    else:
        print(f"\n[2/5] Computing node entropy ({conf.get('entropy_runs', 3)} runs)...")
        # Compute class weights for balanced entropy training
        train_labels_ent = labels[train_mask]
        class_counts_ent = torch.bincount(train_labels_ent, minlength=n_class).float()
        cw_ent = (1.0 / class_counts_ent.clamp(min=1))
        cw_ent = (cw_ent / cw_ent.sum() * n_class).to(device)
        # Pass adj directly (sparse OK, normalize_adj handles it)
        entropy = compute_entropy_standalone(
            features, adj, labels, train_mask, val_mask,
            n_feat, n_class, device,
            n_runs=conf.get('entropy_runs', 3),
            n_epochs=conf.get('entropy_epochs', 200),
            lr=conf.get('entropy_lr', 0.01),
            hidden=conf.get('n_hidden', 64),
            dropout=conf.get('dropout', 0.5),
            class_weights=cw_ent,
        )
        torch.save(entropy, entropy_path)
        print(f"  Entropy saved to {entropy_path}")

    # Free GPU memory before model creation
    torch.cuda.empty_cache()
    print(f"  Entropy: mean={entropy.mean():.4f}, std={entropy.std():.4f}")

    # Re-seed with run-specific seed after entropy computation
    # This ensures different runs produce different results
    run_seed = conf.get('_run_seed', conf.get('seed', 42))
    set_seed(run_seed)

    # --- Build model (dynamic backbone dispatch) ---
    # NOTE: diffusion paths have been removed; retain flag as False for
    # backward-compat with any remaining dead code referring to use_diffusion.
    use_diffusion = False
    backbone = conf.get('backbone', 'GRCN').upper()
    model_name = f"Fair{backbone}"
    print(f"\n[3/5] Building {model_name} model (backbone={backbone})...")

    if backbone == 'GRCN':
        from models.GRCN.fair_grcn import FairGRCN
        model = FairGRCN(
            n_nodes=n_nodes, n_feat=n_feat, n_class=n_class,
            entropy=entropy, sensitive=sensitive, conf=conf, device=device,
        ).to(device)
    elif backbone == 'PROGNN':
        from models.PROGNN.fair_prognn import FairPROGNN
        model = FairPROGNN(
            n_nodes=n_nodes, n_feat=n_feat, n_class=n_class,
            entropy=entropy, sensitive=sensitive, conf=conf, device=device,
        ).to(device)
    elif backbone == 'PROSE':
        from models.PROSE.fair_prose import FairPROSE
        model = FairPROSE(
            n_nodes=n_nodes, n_feat=n_feat, n_class=n_class,
            entropy=entropy, sensitive=sensitive, conf=conf, device=device,
        ).to(device)
    elif backbone == 'IDGL':
        from models.IDGL.fair_idgl import FairIDGL
        model = FairIDGL(
            n_nodes=n_nodes, n_feat=n_feat, n_class=n_class,
            entropy=entropy, sensitive=sensitive, conf=conf, device=device,
        ).to(device)
    elif backbone == 'SLAPS':
        from models.SLAPS.fair_slaps import FairSLAPS
        model = FairSLAPS(
            n_nodes=n_nodes, n_feat=n_feat, n_class=n_class,
            entropy=entropy, sensitive=sensitive, conf=conf, device=device,
        ).to(device)
    else:
        raise NotImplementedError(
            f"Backbone '{backbone}' not yet implemented. "
            f"Available in train.py: GRCN, PROGNN, PROSE, IDGL, SLAPS. "
            f"Use train_sublime.py for SUBLIME."
        )

    # Optimizers (4 base + 1 optional diffusion)
    optim_base = torch.optim.Adam(
        model.base_parameters(),
        lr=conf.get('lr', 0.01),
        weight_decay=conf.get('weight_decay', 5e-4),
    )
    optim_graph = torch.optim.Adam(
        model.graph_parameters(),
        lr=conf.get('lr_graph', 0.01),
    )
    optim_ungsl = torch.optim.Adam(
        model.ungsl_parameters(),
        lr=conf.get('ungsl_lr', 0.001),
    )
    optim_fair = torch.optim.Adam(
        model.fairness_parameters(),
        lr=conf.get('fair_lr', 0.001),
    )

    # Adversarial discriminator (v5: backbone-agnostic fairness)
    adv_weight = conf.get('adv_weight', 0.0)  # 0 = disabled (backward compat)
    discriminator = None
    optim_disc = None
    if adv_weight > 0:
        n_hidden = conf.get('n_hidden', 64)
        discriminator = SensitiveDiscriminator(n_hidden).to(device)
        optim_disc = torch.optim.Adam(
            discriminator.parameters(),
            lr=conf.get('disc_lr', 0.001),
        )
        print(f"  Adversarial debiasing enabled (weight={adv_weight})")

    # Fair Contrastive Learning (v8: sensitive-aware SupCon)
    fair_cl_weight = conf.get('fair_cl_weight', 0.0)  # 0 = disabled
    fair_cl_head = None
    optim_fcl = None
    if fair_cl_weight > 0:
        n_hidden = conf.get('n_hidden', 64)
        fair_cl_head = FairContrastiveHead(
            hidden_dim=n_hidden,
            proj_dim=conf.get('fair_cl_proj_dim', 64),
            temperature=conf.get('fair_cl_temperature', 0.5),
            mode=conf.get('fair_cl_mode', 'supcon_sens'),
        ).to(device)
        optim_fcl = torch.optim.Adam(
            fair_cl_head.parameters(),
            lr=conf.get('fair_cl_lr', 0.001),
        )
        print(f"  Fair Contrastive enabled (weight={fair_cl_weight}, mode={conf.get('fair_cl_mode','supcon_sens')}, "
              f"蟿={conf.get('fair_cl_temperature',0.5)})")

    # 5th optimizer for diffusion parameters
    optim_diff = None
    if use_diffusion:
        optim_diff = torch.optim.Adam(
            model.diffusion_parameters(),
            lr=conf.get('diff_lr', 0.005),
        )
    # 6th optimizer for contrastive learning
    optim_cl = None
    cl_weight = 0.0
    if use_diffusion and hasattr(model, 'contrastive') and model.contrastive is not None:
        optim_cl = torch.optim.Adam(
            model.contrastive.parameters_list(),
            lr=conf.get('cl_lr', 0.001),
        )
        cl_weight = conf.get('cl_weight', 0.5)
        print(f"  Contrastive learning: cl_lr={conf.get('cl_lr', 0.001)}, cl_weight={cl_weight}")
    # PROSE contrastive learning optimizer
    if backbone == 'PROSE' and hasattr(model, 'contrastive_parameters'):
        cl_params = model.contrastive_parameters()
        if cl_params:
            optim_cl = torch.optim.Adam(cl_params, lr=conf.get('lr_cl', 0.01),
                                         weight_decay=conf.get('w_decay_cl', 0.0))
            print(f"  PROSE contrastive: lr_cl={conf.get('lr_cl', 0.01)}")
    # Adversarial (kept for compatibility, disabled by default)
    optim_adv = None
    adv_weight = 0.0

    # Fairness loss
    fair_loss_fn = FairnessLoss(
        alpha_sp=conf.get('fair_alpha_sp', 1.0),
        alpha_struct=conf.get('fair_alpha_struct', 0.5),
        alpha_max=conf.get('fair_alpha_max', 1.0),
        warmup_epochs=conf.get('fair_warmup_epochs', 50),
        alpha_eo=conf.get('fair_alpha_eo', 0.0),
    )

    # Task loss with class weighting (fixes majority class bias)
    # class_weight_strength: 0.0 = uniform, 1.0 = full inverse frequency
    cw_strength = conf.get('class_weight_strength', 1.0)
    train_labels = labels[train_mask]
    class_counts = torch.bincount(train_labels, minlength=n_class).float()
    inv_freq = (1.0 / class_counts.clamp(min=1)).cpu()
    inv_freq = inv_freq / inv_freq.sum() * n_class  # normalize to sum=n_class
    uniform = torch.ones(n_class)
    class_weights = ((1 - cw_strength) * uniform + cw_strength * inv_freq).to(device)
    print(f"  Class weights (strength={cw_strength}): {class_weights.cpu().tolist()}")
    task_loss_fn = lambda output, target: F.cross_entropy(output, target, weight=class_weights)

    # --- Training ---
    n_epochs = conf.get('n_epochs', 200)
    model_selection = conf.get('model_selection', 'algorithm1')
    print(f"\n[4/5] Training for {n_epochs} epochs (selection: {model_selection})...")

    best_val_acc = 0.0
    best_state = None
    best_result = None
    patience = conf.get('patience', 100)
    patience_counter = 0
    epoch_history = []

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()

        # Zero gradients
        optim_base.zero_grad()
        optim_graph.zero_grad()
        optim_ungsl.zero_grad()
        if optim_diff is not None:
            optim_diff.zero_grad()
        if optim_cl is not None:
            optim_cl.zero_grad()
        optim_fair.zero_grad()
        if optim_disc is not None:
            optim_disc.zero_grad()
        if optim_fcl is not None:
            optim_fcl.zero_grad()

        # Forward pass
        output, adj_new, adj_final = model(features, adj)

        # Task loss
        loss_task = task_loss_fn(output[train_mask], labels[train_mask])

        # Fairness loss (structure-level)
        # Mode F: use 蠁-path output for fairness loss (gradient 鈫?蠁 + GCN shared weights)
        # GCN weights are shared between main and 蠁 paths, so fairness signal
        # propagates to main output through shared parameters.
        output_for_fair = output
        loss_task_fair = torch.tensor(0.0, device=device)
        if conf.get('fair_mode') == 'F' and hasattr(model, '_output_fair_cache') and model._output_fair_cache is not None:
            output_for_fair = model._output_fair_cache

        if adj_new.dim() == 2 and adj_new.shape[0] == adj_new.shape[1]:
            loss_fair = fair_loss_fn(output_for_fair, adj_new, sensitive, train_mask, epoch, labels=labels)
        else:
            loss_fair = fair_loss_fn(output_for_fair, None, sensitive, train_mask, epoch, labels=labels)

        # Contrastive loss (improves AUC via better representations)
        # Note: PROSE handles contrastive loss separately via model.compute_contrastive_loss()
        loss_cl = torch.tensor(0.0, device=device)
        if backbone != 'PROSE' and optim_cl is not None and hasattr(model, 'contrastive') and model.contrastive is not None:
            if isinstance(adj_final, tuple):
                indices, values, shape = adj_final
                adj_fair_norm = normalize_adj_sparse(
                    torch.sparse_coo_tensor(indices, values.detach(), shape).coalesce())
            elif adj_final.is_sparse:
                adj_fair_norm = normalize_adj_sparse(adj_final.detach())
            else:
                adj_fair_norm = normalize_adj(adj_final.detach())

            adj_orig_norm = model._adj_norm_cache
            loss_cl = cl_weight * model.contrastive(
                model.classifier, features, adj_fair_norm, adj_orig_norm, train_mask
            )

        # PROGNN graph regularization (Frobenius + L1 + nuclear + smoothness)
        if backbone == 'PROGNN':
            loss_graph_reg = model.compute_graph_reg_loss()
        else:
            loss_graph_reg = torch.tensor(0.0, device=device)

        # PROSE contrastive loss + degree regularization
        loss_prose = torch.tensor(0.0, device=device)
        if backbone == 'PROSE':
            loss_prose = model.compute_contrastive_loss(features) + model.get_degree_loss()

        # IDGL iterative graph regularization + Mode G graph fairness loss
        loss_idgl = torch.tensor(0.0, device=device)
        if backbone == 'IDGL':
            loss_idgl = model.get_iter_loss()
            if hasattr(model, 'get_graph_fair_loss'):
                gf_weight = conf.get('graph_fair_weight', 0.0)
                if gf_weight > 0:
                    loss_idgl = loss_idgl + gf_weight * model.get_graph_fair_loss()

        # SLAPS DAE loss (feature reconstruction self-supervision)
        # Joint training: DAE + task loss from epoch 1 (simpler than paper's two-phase)
        # The small dae_weight keeps DAE as auxiliary regularization, not dominant.
        loss_slaps = torch.tensor(0.0, device=device)
        if backbone == 'SLAPS':
            dae_weight = conf.get('dae_weight', 0.01)
            loss_slaps = dae_weight * model.get_dae_loss(features)

        # Adversarial debiasing loss (v5)
        loss_adv = torch.tensor(0.0, device=device)
        if discriminator is not None and hasattr(model.classifier, '_hidden'):
            adv_alpha = fair_loss_fn.get_alpha(epoch)  # Same curriculum as fairness loss
            loss_adv = adv_weight * compute_adv_loss(
                discriminator, model.classifier._hidden, sensitive, train_mask, alpha=adv_alpha)

        # v7: Counterfactual consistency loss
        # Force predictions to be invariant to sensitive attribute perturbation
        loss_cf = torch.tensor(0.0, device=device)
        if cf_features is not None and cf_weight > 0:
            cf_alpha = fair_loss_fn.get_alpha(epoch)  # Same warmup schedule
            output_cf, _, _ = model(cf_features, adj)
            # MSE between predictions on original and counterfactual (training nodes only)
            if output.dim() == 1:
                loss_cf = cf_weight * cf_alpha * F.mse_loss(output[train_mask], output_cf[train_mask])
            else:
                probs = F.softmax(output, dim=1)
                probs_cf = F.softmax(output_cf, dim=1)
                loss_cf = cf_weight * cf_alpha * F.mse_loss(probs[train_mask], probs_cf[train_mask])

        # v8: Fair Supervised Contrastive Loss (Sensitive-aware SupCon)
        # Pulls same-label-different-sensitive nodes together,
        # pushes apart same-label-same-sensitive (avoid group homogeneity)
        loss_fcl = torch.tensor(0.0, device=device)
        if fair_cl_head is not None and fair_cl_weight > 0:
            # Access hidden representation from backbone
            h_for_cl = None
            if hasattr(model, 'classifier') and hasattr(model.classifier, '_hidden'):
                h_for_cl = model.classifier._hidden
            elif hasattr(model, '_last_hidden'):
                h_for_cl = model._last_hidden
            if h_for_cl is not None:
                fcl_alpha = fair_loss_fn.get_alpha(epoch)  # Curriculum warmup
                loss_fcl = fair_cl_weight * fcl_alpha * fair_cl_head(
                    h_for_cl, labels, sensitive, train_mask)

        # Total loss
        loss = (loss_task + loss_fair + loss_task_fair + loss_adv + loss_cl
                + loss_graph_reg + loss_prose + loss_idgl + loss_cf + loss_fcl + loss_slaps)

        # Backward
        loss.backward()

        # Gradient clipping for PROGNN (prevents FGP.Adj from diverging)
        if backbone == 'PROGNN':
            torch.nn.utils.clip_grad_norm_(model.graph_parameters(), max_norm=1.0)

        # Optimizer steps
        optim_base.step()
        optim_graph.step()
        optim_ungsl.step()
        optim_fair.step()
        if optim_disc is not None:
            optim_disc.step()
        if optim_diff is not None:
            optim_diff.step()
        if optim_cl is not None:
            optim_cl.step()
        if optim_fcl is not None:
            optim_fcl.step()

        # PROGNN: project adjacency to [0, 1] after gradient step
        if backbone == 'PROGNN':
            with torch.no_grad():
                model.fgp.Adj.data.clamp_(0, 1)

        # PROSE: structure bootstrapping
        if backbone == 'PROSE':
            model.maybe_bootstrap(epoch)

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            output_eval, adj_new_eval, _ = model(features, adj)

        train_metrics = compute_fairness_metrics(
            output_eval.detach(), labels, sensitive, train_mask
        )
        val_metrics = compute_fairness_metrics(
            output_eval.detach(), labels, sensitive, val_mask
        )

        # Record epoch history for Algorithm 1 selection
        if model_selection == 'algorithm1':
            save_interval = conf.get('alg1_save_interval', 5 if n_nodes > 50000 else 1)
            if epoch % save_interval == 0 or epoch == n_epochs - 1:
                epoch_history.append({
                    'epoch': epoch,
                    'acc': val_metrics['acc'],
                    'auc': val_metrics['auc'],
                    'f1': val_metrics['f1'],
                    'sp': val_metrics['sp'],
                    'eo': val_metrics['eo'],
                    'state_dict': {k: v.cpu().clone() for k, v in model.named_parameters()},
                })

        # Legacy tradeoff-based early stopping
        w_auc = conf.get('es_auc_weight', 2.0)
        w_fair = conf.get('es_fair_weight', 1.0)
        tradeoff = val_metrics['acc'] + w_auc * val_metrics['auc'] + val_metrics['f1'] \
                   - w_fair * (val_metrics['sp'] + val_metrics['eo'])
        improve = ''

        if tradeoff > best_val_acc:
            improve = '*'
            best_val_acc = tradeoff
            if model_selection in ('tradeoff', 'best_val_acc'):
                best_state = deepcopy(model.state_dict())
            best_result = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Fallback: also save best state based on pure val accuracy (for 'best_val_acc' mode)
        # This guards against best_state being None at end of training.
        if model_selection == 'best_val_acc' and best_state is None:
            best_state = deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or improve:
            if hasattr(model.fair_ungsl, 'lambdas'):
                lambda_mean = torch.sigmoid(model.fair_ungsl.lambdas).mean().item()
            elif hasattr(model.fair_ungsl, 'delta_eps_learnable'):
                lambda_mean = model.fair_ungsl.delta_eps_learnable.mean().item()
            else:
                lambda_mean = 0.0
            with torch.no_grad():
                val_preds = output_eval[val_mask].argmax(dim=1) if output_eval.dim() > 1 else (output_eval[val_mask] > 0).long()
                pred_dist = torch.bincount(val_preds, minlength=n_class)
                pred_str = '/'.join([str(int(pred_dist[c])) for c in range(n_class)])
            print(
                f"  Epoch {epoch + 1:4d} | "
                f"Loss {loss.item():.4f} | "
                f"Acc {val_metrics['acc']:.2f}% "
                f"AUC {val_metrics['auc']:.2f}% "
                f"F1 {val_metrics['f1']:.2f}% | "
                f"SP {val_metrics['sp']:.2f}% "
                f"EO {val_metrics['eo']:.2f}% | "
                f"Pred {pred_str} "
                f"位虅={lambda_mean:.3f} {improve}"
            )

        # Patience-based stopping only in tradeoff mode
        if model_selection == 'tradeoff' and patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # --- Model Selection ---
    if model_selection == 'algorithm1':
        selected = algorithm1_select(epoch_history)
        best_state = selected['state_dict']
        print(f"\n  [Algorithm 1] Selected epoch {selected['epoch']+1}/{n_epochs}")
        print(f"    Val: ACC={selected['acc']:.2f}% AUC={selected['auc']:.2f}% "
              f"F1={selected['f1']:.2f}% | SP={selected['sp']:.2f}% EO={selected['eo']:.2f}%")
        epoch_history.clear()

    # --- Test ---
    print(f"\n[5/5] Testing...")
    model.load_state_dict(best_state, strict=False)
    model.eval()

    with torch.no_grad():
        output_test, _, _ = model(features, adj)

    test_metrics = compute_fairness_metrics(
        output_test.detach(), labels, sensitive, test_mask
    )

    # Per-class prediction distribution on test set
    with torch.no_grad():
        test_preds = output_test[test_mask].argmax(dim=1) if output_test.dim() > 1 else (output_test[test_mask] > 0).long()
        test_pred_dist = torch.bincount(test_preds, minlength=n_class)
        test_true_dist = torch.bincount(labels[test_mask], minlength=n_class)

    print(f"\n{'=' * 70}")
    print(f"  {dataset_name} Results:")
    print(f"  Acc={test_metrics['acc']:.2f}%  AUC={test_metrics['auc']:.2f}%  F1={test_metrics['f1']:.2f}%  |  SP={test_metrics['sp']:.2f}%  EO={test_metrics['eo']:.2f}%")
    print(f"  Test true dist: {test_true_dist.cpu().tolist()}, pred dist: {test_pred_dist.cpu().tolist()}")
    # v3 diagnostics: print 尾_s1 and 渭 if present
    if hasattr(model.fair_ungsl, 'beta_minority'):
        with torch.no_grad():
            b_s1_eff = model.fair_ungsl._beta_s1().item()
        bm_raw = model.fair_ungsl.beta_minority.item()
        b0 = model.fair_ungsl.Beta
        mu = getattr(model.fair_ungsl, 'mu', None)
        constrained = getattr(model.fair_ungsl, 'beta_minority_constrained', False)
        print(f"  [v3 diag] beta_s0 (fixed)={b0:.4f}  beta_s1 (effective)={b_s1_eff:+.4f}  "
              f"beta_s1 (raw)={bm_raw:+.4f}  mu={mu}  constrained={constrained}")
    print(f"{'=' * 70}")

    test_metrics['model_selection'] = model_selection
    if model_selection == 'algorithm1':
        test_metrics['selected_epoch'] = selected['epoch'] + 1

    # === dump extra metrics patch ===
    _dump_path = conf.get('dump_extra_path')
    if _dump_path:
        try:
            from measure_utils import dump_extra_metrics
            _extra = dump_extra_metrics(model, features, adj, entropy, sensitive, _dump_path, conf=conf)
            if _extra:
                test_metrics.update(_extra)
        except Exception as _e:
            print(f"  [dump_extra_metrics] skipped: {_e}")
    # === end patch ===

    return test_metrics


# ============================================================
# Main
# ============================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='FairUnGSL Training')
    parser.add_argument('--dataset', type=str, default='german',
                        choices=['german', 'germanA', 'bail', 'bailA', 'credit', 'creditA', 'pokec_z', 'pokec_n', 'nba', 'syn1', 'syn2', 'sport', 'occupation'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='dataset/',
                        help='Root directory for datasets')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['GRCN', 'PROGNN', 'PROSE', 'IDGL', 'SLAPS', 'SUBLIME'],
                        # IDGL uses iterative graph refinement with weighted cosine similarity
                        help='GSL backbone (default: inferred from config or GRCN)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of experiment runs')

    # Model hyperparameters (can be overridden by config file)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_graph', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--k', type=int, default=20,
                        help='KNN neighbors for graph learning')
    parser.add_argument('--fusion_ratio', type=float, default=0.5,
                        help='Weight for learned graph in fusion')

    # UnGSL parameters
    parser.add_argument('--init_value', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--ungsl_lr', type=float, default=0.001)

    # Fairness parameters
    parser.add_argument('--lambda_init', type=float, default=0.0)
    parser.add_argument('--fair_lr', type=float, default=0.001)
    parser.add_argument('--fair_alpha_sp', type=float, default=1.0)
    parser.add_argument('--fair_alpha_struct', type=float, default=0.5)
    parser.add_argument('--fair_alpha_max', type=float, default=1.0)
    parser.add_argument('--fair_warmup_epochs', type=int, default=50)

    # Entropy
    parser.add_argument('--entropy_runs', type=int, default=3)
    parser.add_argument('--entropy_epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=100)

    args = parser.parse_args()

    # Build config dict
    conf = vars(args).copy()

    # Override with config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_conf = json.load(f)
        conf.update(file_conf)
        print(f"Loaded config from {args.config}")

    # CLI --backbone overrides config-file backbone if both given; default GRCN.
    # Note: conf already has 'backbone' key from vars(args), possibly None.
    if args.backbone is not None:
        conf['backbone'] = args.backbone
    if conf.get('backbone') is None:
        conf['backbone'] = 'GRCN'


    # Device
    if not torch.cuda.is_available() and 'cuda' in conf['device']:
        print("CUDA not available, using CPU")
        conf['device'] = 'cpu'
    device = conf['device']

    # Run experiments
    all_results = []
    for run in range(conf['n_runs']):
        print(f"\n{'#' * 60}")
        print(f"# Run {run + 1}/{conf['n_runs']}")
        print(f"{'#' * 60}")

        run_seed = conf['seed'] + run * 1000
        set_seed(run_seed)
        conf['_run_seed'] = run_seed  # pass to train_fair_grcn
        result = train_fair_grcn(conf, device)
        all_results.append(result)

    # Summary
    metrics_keys = ['acc', 'auc', 'f1', 'sp', 'eo']
    summary = {}
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: FairUnGSL on {conf['dataset']} ({conf['n_runs']} runs)")
    print(f"  {'Metric':<8} {'Mean':>8} {'卤 Std':>8}")
    print(f"  {'-'*26}")
    for k in metrics_keys:
        vals = [r[k] for r in all_results]
        m, s = np.mean(vals), np.std(vals)
        summary[k] = {'mean': float(m), 'std': float(s)}
        arrow = '鈫? if k in ['acc', 'auc', 'f1'] else '鈫?
        print(f"  {k.upper():<5}{arrow}  {m:>7.2f}% 卤 {s:.2f}%")
    print(f"{'=' * 70}")

    # Save results
    results_dir = conf.get('results_dir', 'results/')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"FairUnGSL_{conf.get('backbone','GRCN').lower()}_{conf['dataset']}_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'config': {k: v for k, v in conf.items() if isinstance(v, (int, float, str, bool))},
            'runs': all_results,
            'summary': summary,
        }, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
