"""
measure_utils.py - Helpers for minority-label sensitivity experiments.

Provides:
1. apply_minority_label_ratio: stratified subsampling of g=1 training labels.
2. extract_edge_metrics: per-group-pair median edge weight, active count, pass rate.
3. extract_entropy_metrics: mean entropy by group + entropy gap.
4. dump_extra_metrics: end-of-training dump of all secondary metrics to JSON.

These functions are imported by train.py only when conf has the relevant fields,
so existing experiments are unaffected.
"""
import json
import os
import numpy as np
import torch


def apply_minority_label_ratio(idx_train, sensitive, labels, ratio, seed):
    """
    Reduce g=1 (minority) training indices to `ratio` fraction, stratified by label.
    g=0 (majority) indices are kept unchanged. validation/test indices not touched.
    """
    if ratio >= 0.9999:
        return idx_train, {
            "n_train_g0": int(((sensitive[idx_train] == 0)).sum().item()),
            "n_train_g1_kept": int(((sensitive[idx_train] == 1)).sum().item()),
            "n_train_g1_dropped": 0,
            "ratio_actual": 1.0,
        }
    idx_train_arr = np.asarray(idx_train.cpu() if torch.is_tensor(idx_train) else idx_train, dtype=np.int64)
    sens_arr = sensitive.cpu().numpy() if torch.is_tensor(sensitive) else np.asarray(sensitive)
    labs_arr = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    g0_mask = sens_arr[idx_train_arr] == 0
    g1_mask = sens_arr[idx_train_arr] == 1
    idx_train_g0 = idx_train_arr[g0_mask]
    idx_train_g1 = idx_train_arr[g1_mask]

    rng = np.random.default_rng(seed)
    kept_g1_list = []
    for c in sorted(np.unique(labs_arr)):
        c = int(c)
        idx_g1_c = idx_train_g1[labs_arr[idx_train_g1] == c]
        if len(idx_g1_c) == 0:
            continue
        n_keep = max(1, int(round(len(idx_g1_c) * ratio)))
        n_keep = min(n_keep, len(idx_g1_c))
        chosen = rng.choice(idx_g1_c, size=n_keep, replace=False)
        kept_g1_list.append(chosen)
    kept_g1 = np.concatenate(kept_g1_list) if kept_g1_list else np.array([], dtype=np.int64)

    new_idx = np.concatenate([idx_train_g0, kept_g1])
    new_idx.sort()
    info = {
        "n_train_g0": int(len(idx_train_g0)),
        "n_train_g1_kept": int(len(kept_g1)),
        "n_train_g1_dropped": int(len(idx_train_g1) - len(kept_g1)),
        "ratio_actual": float(len(kept_g1) / max(1, len(idx_train_g1))),
    }

    if torch.is_tensor(idx_train):
        new_idx_train = torch.from_numpy(new_idx).to(idx_train.device)
    else:
        new_idx_train = new_idx
    return new_idx_train, info


def _coalesce_sparse_to_arrays(adj):
    if isinstance(adj, tuple):
        indices, values, shape = adj
    elif adj.is_sparse:
        adj = adj.coalesce()
        indices = adj.indices()
        values = adj.values()
    else:
        nz = adj.nonzero(as_tuple=False)
        indices = nz.t()
        values = adj[nz[:, 0], nz[:, 1]]
    src = indices[0].cpu().numpy()
    dst = indices[1].cpu().numpy()
    vals = values.detach().cpu().numpy()
    return src, dst, vals


def extract_edge_metrics(adj, sensitive, gate_floor=0.1, atol=1e-3):
    """Compute per-group-pair edge weight statistics."""
    src, dst, vals = _coalesce_sparse_to_arrays(adj)
    s = sensitive.cpu().numpy() if torch.is_tensor(sensitive) else np.asarray(sensitive)
    s_src, s_dst = s[src], s[dst]
    pair_majmaj = (s_src == 0) & (s_dst == 0)
    pair_minmin = (s_src == 1) & (s_dst == 1)
    pair_cross = s_src != s_dst

    def _med(mask):
        v = vals[mask]
        return float(np.median(v)) if len(v) > 0 else float("nan")

    def _active(mask):
        v = vals[mask]
        return int((v > gate_floor + atol).sum()), int(len(v))

    med_maj, med_min, med_cross = _med(pair_majmaj), _med(pair_minmin), _med(pair_cross)
    act_maj_n, tot_maj = _active(pair_majmaj)
    act_min_n, tot_min = _active(pair_minmin)
    act_cross_n, tot_cross = _active(pair_cross)

    def _safe_div(a, b):
        if isinstance(a, float) and (a != a or b == 0 or (isinstance(b, float) and b != b)):
            return float("nan")
        return float(a / b)

    return {
        "median_majmaj": med_maj,
        "median_minmin": med_min,
        "median_cross": med_cross,
        "cross_over_maj": _safe_div(med_cross, med_maj),
        "min_over_maj": _safe_div(med_min, med_maj),
        "n_edges_majmaj": tot_maj,
        "n_edges_minmin": tot_min,
        "n_edges_cross": tot_cross,
        "n_active_majmaj": act_maj_n,
        "n_active_minmin": act_min_n,
        "n_active_cross": act_cross_n,
        "pass_rate_majmaj": _safe_div(act_maj_n, tot_maj),
        "pass_rate_minmin": _safe_div(act_min_n, tot_min),
        "pass_rate_cross": _safe_div(act_cross_n, tot_cross),
    }


def extract_entropy_metrics(entropy, sensitive):
    """Compute mean entropy by group + gap."""
    e = entropy.detach().cpu().numpy() if torch.is_tensor(entropy) else np.asarray(entropy)
    s = sensitive.cpu().numpy() if torch.is_tensor(sensitive) else np.asarray(sensitive)
    e0 = e[s == 0]
    e1 = e[s == 1]
    mean_g0 = float(e0.mean()) if len(e0) else float("nan")
    mean_g1 = float(e1.mean()) if len(e1) else float("nan")
    return {
        "entropy_g0": mean_g0,
        "entropy_g1": mean_g1,
        "entropy_gap": mean_g1 - mean_g0,
    }


def dump_extra_metrics(model, features, adj, entropy, sensitive, save_path, conf=None):
    """Forward once and dump all metrics as JSON."""
    model.eval()
    with torch.no_grad():
        try:
            output, adj_new, adj_final = model(features, adj)
        except Exception as e:
            print(f"[dump_extra_metrics] forward failed: {e}")
            return None

    target_adj = adj_new if adj_new is not None else adj_final

    try:
        ent_metrics = extract_entropy_metrics(entropy, sensitive)
    except Exception as e:
        print(f"[dump_extra_metrics] entropy extract failed: {e}")
        ent_metrics = {"entropy_g0": float("nan"), "entropy_g1": float("nan"), "entropy_gap": float("nan")}

    try:
        gate_floor = float(conf.get("beta", 0.1)) if conf else 0.1
        edge_metrics = extract_edge_metrics(target_adj, sensitive, gate_floor=gate_floor)
    except Exception as e:
        print(f"[dump_extra_metrics] edge extract failed: {e}")
        edge_metrics = {}

    record = {}
    record.update(ent_metrics)
    record.update(edge_metrics)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    gap = ent_metrics.get("entropy_gap")
    gap_s = f"{gap:.4f}" if isinstance(gap, float) and gap == gap else "nan"
    print(f"  [extra_metrics] dumped to {save_path}: gap={gap_s} "
          f"med_maj={edge_metrics.get('median_majmaj')} med_min={edge_metrics.get('median_minmin')} "
          f"med_cross={edge_metrics.get('median_cross')} cross_over_maj={edge_metrics.get('cross_over_maj')}")
    return record
