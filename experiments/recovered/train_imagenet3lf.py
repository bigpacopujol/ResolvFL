import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import os, json, time

# ============================================================
# CONFIG
# ============================================================
FEAT_DIR = "."          # en Colab: "/content/drive/MyDrive/features_dino"
CKPT_DIR = "."           # en Colab: "/content/drive/MyDrive/imagenet3lf_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

QUICK_TEST = False        # en el test local lo forzamos con datos ya pequeños
SEEDS = [1, 2, 3]
K = 10
ALPHA_CLS = 0.5
ALPHA_DEP = 0.5
N_ROUNDS = 50
EVAL_EVERY = 10
LOCAL_EPOCHS = 1
LR = 0.05
BATCH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ============================================================
# 1. Cargar datos
# ============================================================
Xtr = np.load(os.path.join(FEAT_DIR, "Xtr.npy"), mmap_mode="r")
Ltr = np.load(os.path.join(FEAT_DIR, "Ltr.npy"))
Xva = np.load(os.path.join(FEAT_DIR, "Xva.npy"), mmap_mode="r")
Lva = np.load(os.path.join(FEAT_DIR, "Lva.npy"))
print(f"train: {Xtr.shape}  val: {Xva.shape}")

# ============================================================
# 2. Taxonomia con IDs compactos (misma logica ya validada)
# ============================================================
basic_raw, basic_local = np.unique(Ltr[:, 0], return_inverse=True)
subord_key = basic_local.astype(np.int64) * 100000 + Ltr[:, 1]
subord_uniq, subord_local = np.unique(subord_key, return_inverse=True)
leaf_key = subord_local.astype(np.int64) * 1000000 + Ltr[:, 2]
leaf_uniq, leaf_local = np.unique(leaf_key, return_inverse=True)

N_BASIC, N_SUBORD, N_LEAF = len(basic_raw), len(subord_uniq), len(leaf_uniq)
print(f"basic={N_BASIC} subord={N_SUBORD} leaf={N_LEAF}")

subord_parent = np.zeros(N_SUBORD, dtype=np.int64)
for bl, sl in zip(basic_local, subord_local):
    subord_parent[sl] = bl
leaf_parent_local = np.zeros(N_LEAF, dtype=np.int64)
for sl, ll in zip(subord_local, leaf_local):
    leaf_parent_local[ll] = sl

children_per_basic = np.bincount(subord_parent, minlength=N_BASIC)
children_per_subord = np.bincount(leaf_parent_local, minlength=N_SUBORD)
assert children_per_basic.min() > 0 and children_per_subord.min() > 0

class_raw_to_leaf = {}
for cr, ll in zip(Ltr[:, 2].tolist(), leaf_local.tolist()):
    class_raw_to_leaf[cr] = ll
val_leaf_local = np.array([class_raw_to_leaf[c] for c in Lva[:, 2].tolist()], dtype=np.int64)

basic_child_index = {}
for b in range(N_BASIC):
    subs = np.where(subord_parent == b)[0]
    basic_child_index[b] = {s: i for i, s in enumerate(subs)}
subord_child_index = {}
for s in range(N_SUBORD):
    leaves = np.where(leaf_parent_local == s)[0]
    subord_child_index[s] = {l: i for i, l in enumerate(leaves)}

def local_targets(basic_arr, subord_arr, leaf_arr):
    t_root = basic_arr
    t_basic = np.array([basic_child_index[b][s] for b, s in zip(basic_arr, subord_arr)])
    t_subord = np.array([subord_child_index[s][l] for s, l in zip(subord_arr, leaf_arr)])
    return t_root, t_basic, t_subord

# flat: espacio de nodos unificado (offsets)
FLAT_OFFSET_BASIC, FLAT_OFFSET_SUBORD, FLAT_OFFSET_LEAF = 0, N_BASIC, N_BASIC + N_SUBORD
N_FLAT = N_BASIC + N_SUBORD + N_LEAF
IS_LEAF_FLAT = np.zeros(N_FLAT, dtype=bool)
IS_LEAF_FLAT[FLAT_OFFSET_LEAF:] = True
print(f"N_FLAT={N_FLAT} (basic {N_BASIC} + subord {N_SUBORD} + leaf {N_LEAF})")

Xtr_full = np.asarray(Xtr)
Xva_t = torch.tensor(np.asarray(Xva), dtype=torch.float32)

# ============================================================
# 3. Particion federada (una por semilla, guardada para post-hoc)
# ============================================================
def make_partition(seed):
    rng = np.random.default_rng(seed)
    client_of_sample = np.zeros(N_LEAF and len(leaf_local), dtype=np.int64)
    for c in range(N_LEAF):
        idx_c = np.where(leaf_local == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([ALPHA_CLS] * K)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx_c, cuts)):
            client_of_sample[part] = k
    nu = rng.dirichlet([ALPHA_DEP] * 3, size=K)
    depth_of_sample = np.zeros(len(leaf_local), dtype=np.int64)
    for k in range(K):
        idx_k = np.where(client_of_sample == k)[0]
        depth_of_sample[idx_k] = rng.choice([1, 2, 3], size=len(idx_k), p=nu[k])
    return client_of_sample, depth_of_sample

# ============================================================
# 4. Modelos
# ============================================================
class TreeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.root = nn.Linear(2048, N_BASIC)
        self.basic_clf = nn.ModuleList([nn.Linear(2048, children_per_basic[b]) for b in range(N_BASIC)])
        self.subord_clf = nn.ModuleList([nn.Linear(2048, children_per_subord[s]) for s in range(N_SUBORD)])

class FlatHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, N_FLAT)

def flat_target(basic_arr, subord_arr, leaf_arr, depth_arr):
    t = np.zeros(len(depth_arr), dtype=np.int64)
    m1 = depth_arr == 1
    m2 = depth_arr == 2
    m3 = depth_arr == 3
    t[m1] = FLAT_OFFSET_BASIC + basic_arr[m1]
    t[m2] = FLAT_OFFSET_SUBORD + subord_arr[m2]
    t[m3] = FLAT_OFFSET_LEAF + leaf_arr[m3]
    return t

# ============================================================
# 5. Entrenamiento (tree: masked/unmasked comparten esta funcion)
# ============================================================
def train_tree(masked, seed, basic_s, subord_s, leaf_s, depth_s, client_of_sample,
               Xtr_t, ckpt_tag, resume=True):
    ckpt_path = os.path.join(CKPT_DIR, f"imagenet3lf_tree_{ckpt_tag}_seed{seed}.pt")
    log_path = os.path.join(CKPT_DIR, f"imagenet3lf_tree_{ckpt_tag}_seed{seed}_log.json")

    torch.manual_seed(seed)  # misma inicializacion para masked y unmasked en esta semilla
    global_head = TreeHead().to(device)
    start_round = 0
    log = {"rounds": [], "leaf_acc": [], "per_node_acc": []}

    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        global_head.load_state_dict(state["model"])
        start_round = state["round"]
        if os.path.exists(log_path):
            log = json.load(open(log_path))
        print(f"  [checkpoint] retomando {ckpt_tag} seed{seed} desde ronda {start_round}")

    t_root_all, t_basic_all, t_subord_all = local_targets(basic_s, subord_s, leaf_s)
    client_idx = [np.where(client_of_sample == k)[0] for k in range(K)]

    for rnd in range(start_round, N_ROUNDS):
        t0 = time.time()
        local_states = []
        for k in range(K):
            idx_k = client_idx[k]
            if len(idx_k) == 0:
                local_states.append(None); continue
            local_head = TreeHead().to(device)
            local_head.load_state_dict(global_head.state_dict())
            opt = torch.optim.SGD(local_head.parameters(), lr=LR)
            for _ in range(LOCAL_EPOCHS):
                perm = np.random.permutation(idx_k)
                for i0 in range(0, len(perm), BATCH):
                    bidx = perm[i0:i0 + BATCH]
                    xb = Xtr_t[bidx].to(device)
                    d = depth_s[bidx]
                    logits_root = local_head.root(xb)
                    loss = None
                    m1 = d >= 1
                    if m1.any():
                        l = F.cross_entropy(logits_root[m1], torch.tensor(t_root_all[bidx][m1], device=device))
                        loss = l if loss is None else loss + l
                    m2 = d >= 2
                    if m2.any():
                        for b in np.unique(basic_s[bidx][m2]):
                            sel = m2 & (basic_s[bidx] == b)
                            if sel.sum() == 0: continue
                            lg = local_head.basic_clf[b](xb[sel])
                            l = F.cross_entropy(lg, torch.tensor(t_basic_all[bidx][sel], device=device))
                            loss = l if loss is None else loss + l
                    m3 = d >= 3
                    if m3.any():
                        for s in np.unique(subord_s[bidx][m3]):
                            sel = m3 & (subord_s[bidx] == s)
                            if sel.sum() == 0: continue
                            lg = local_head.subord_clf[s](xb[sel])
                            l = F.cross_entropy(lg, torch.tensor(t_subord_all[bidx][sel], device=device))
                            loss = l if loss is None else loss + l
                    if loss is None:
                        continue
                    opt.zero_grad(); loss.backward(); opt.step()
            local_states.append(local_head.state_dict())

        new_state = {}
        for key in global_head.state_dict().keys():
            parts_, weights_ = [], []
            for k in range(K):
                if local_states[k] is None: continue
                idx_k = client_idx[k]
                d_k = depth_s[idx_k]
                if masked:
                    if "root" in key:
                        w = (d_k >= 1).sum()
                    elif "basic_clf" in key:
                        node_id = int(key.split(".")[1])
                        w = ((d_k >= 2) & (basic_s[idx_k] == node_id)).sum()
                    else:
                        node_id = int(key.split(".")[1])
                        w = ((d_k >= 3) & (subord_s[idx_k] == node_id)).sum()
                    if w == 0: continue
                else:
                    w = len(idx_k)
                weights_.append(w)
                parts_.append(local_states[k][key])
            if not parts_:
                new_state[key] = global_head.state_dict()[key]; continue
            w = torch.tensor(weights_, dtype=torch.float32, device=device)
            w = w / w.sum()
            stacked = torch.stack(parts_)
            new_state[key] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)
        global_head.load_state_dict(new_state)

        torch.save({"model": global_head.state_dict(), "round": rnd + 1}, ckpt_path)

        if (rnd + 1) % EVAL_EVERY == 0 or (rnd + 1) == N_ROUNDS:
            leaf_acc, per_node = eval_tree(global_head)
            log["rounds"].append(rnd + 1)
            log["leaf_acc"].append(leaf_acc)
            log["per_node_acc"].append(per_node)
            json.dump(log, open(log_path, "w"))
            print(f"  [{ckpt_tag} seed{seed}] ronda {rnd+1}/{N_ROUNDS}  "
                  f"leaf_acc={leaf_acc:.2f}%  ({time.time()-t0:.1f}s)")
        else:
            print(f"  [{ckpt_tag} seed{seed}] ronda {rnd+1}/{N_ROUNDS}  ({time.time()-t0:.1f}s)")

    return global_head, log

@torch.no_grad()
def eval_tree(head):
    """Devuelve (leaf_acc global, dict con accuracy por nodo interno)."""
    head.eval()
    correct = 0
    # acumuladores por nodo: {"basic_<id>": [aciertos, total], "subord_<id>": [...]}
    node_hits = defaultdict(lambda: [0, 0])
    all_leaf_pred = np.zeros(len(Xva_t), dtype=np.int64)

    for i0 in range(0, len(Xva_t), 2048):
        xb = Xva_t[i0:i0 + 2048].to(device)
        n = xb.shape[0]
        pb_root = F.softmax(head.root(xb), dim=1)          # (n, N_BASIC)
        pred_root = pb_root.argmax(1).cpu().numpy()
        true_basic = Lva[i0:i0 + 2048, 0]
        true_subord_raw = Lva[i0:i0 + 2048, 1]
        true_leaf_g = val_leaf_local[i0:i0 + 2048]

        best_leaf = np.zeros(n, dtype=np.int64)
        best_p = np.zeros(n, dtype=np.float32)
        for b in range(N_BASIC):
            pbasic = F.softmax(head.basic_clf[b](xb), dim=1).cpu().numpy()  # (n, n_children_b)
            subs = np.where(subord_parent == b)[0]
            for j, s in enumerate(subs):
                pleaf = F.softmax(head.subord_clf[s](xb), dim=1).cpu().numpy()
                leaves = np.where(leaf_parent_local == s)[0]
                for l_idx, leaf_g in enumerate(leaves):
                    p = pb_root[:, b].cpu().numpy() * pbasic[:, j] * pleaf[:, l_idx]
                    better = p > best_p
                    best_p[better] = p[better]
                    best_leaf[better] = leaf_g
        correct += (best_leaf == true_leaf_g).sum()
        all_leaf_pred[i0:i0 + 2048] = best_leaf

        # accuracy por nodo 'basic' (entre las muestras cuyo basic real es b, acierta el root?)
        for b in range(N_BASIC):
            sel = true_basic == b
            if sel.sum() == 0: continue
            node_hits[f"basic_{b}"][0] += (pred_root[sel] == b).sum()
            node_hits[f"basic_{b}"][1] += sel.sum()

        # accuracy por nodo 'subord': mapeamos leaf->subord real usando leaf_parent_local
        true_subord_g = leaf_parent_local[true_leaf_g]
        for s in range(N_SUBORD):
            sel = true_subord_g == s
            if sel.sum() == 0: continue
            b_of_s = subord_parent[s]
            pbasic_s = F.softmax(head.basic_clf[b_of_s](xb[sel]), dim=1).cpu().numpy()
            subs = np.where(subord_parent == b_of_s)[0]
            local_idx = int(np.where(subs == s)[0][0])
            pred_s = pbasic_s.argmax(1)
            node_hits[f"subord_{s}"][0] += (pred_s == local_idx).sum()
            node_hits[f"subord_{s}"][1] += sel.sum()

    leaf_acc = 100.0 * correct / len(Xva_t)
    per_node = {k: (v[0] / v[1] if v[1] > 0 else None) for k, v in node_hits.items()}
    return leaf_acc, per_node

def train_flat(seed, basic_s, subord_s, leaf_s, depth_s, client_of_sample, Xtr_t, resume=True):
    ckpt_path = os.path.join(CKPT_DIR, f"imagenet3lf_flat_seed{seed}.pt")
    torch.manual_seed(seed)
    model = FlatHead().to(device)
    start_round = 0
    if resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        start_round = state["round"]
        print(f"  [checkpoint] retomando flat seed{seed} desde ronda {start_round}")

    targets = flat_target(basic_s, subord_s, leaf_s, depth_s)
    client_idx = [np.where(client_of_sample == k)[0] for k in range(K)]

    for rnd in range(start_round, N_ROUNDS):
        t0 = time.time()
        local_states, weights_ = [], []
        for k in range(K):
            idx_k = client_idx[k]
            if len(idx_k) == 0: continue
            local_model = FlatHead().to(device)
            local_model.load_state_dict(model.state_dict())
            opt = torch.optim.SGD(local_model.parameters(), lr=LR)
            for _ in range(LOCAL_EPOCHS):
                perm = np.random.permutation(idx_k)
                for i0 in range(0, len(perm), BATCH):
                    bidx = perm[i0:i0 + BATCH]
                    xb = Xtr_t[bidx].to(device)
                    yb = torch.tensor(targets[bidx], device=device)
                    logits = local_model.fc(xb)
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
            local_states.append(local_model.state_dict())
            weights_.append(len(idx_k))
        w = torch.tensor(weights_, dtype=torch.float32, device=device)
        w = w / w.sum()
        new_state = {}
        for key in model.state_dict().keys():
            stacked = torch.stack([s[key] for s in local_states])
            new_state[key] = (stacked * w.view(-1, *([1] * (stacked.dim() - 1)))).sum(0)
        model.load_state_dict(new_state)
        torch.save({"model": model.state_dict(), "round": rnd + 1}, ckpt_path)
        print(f"  [flat seed{seed}] ronda {rnd+1}/{N_ROUNDS}  ({time.time()-t0:.1f}s)")

    return model

@torch.no_grad()
def eval_flat(model):
    model.eval()
    correct, abstain = 0, 0
    for i0 in range(0, len(Xva_t), 2048):
        xb = Xva_t[i0:i0 + 2048].to(device)
        logits = model.fc(xb)
        raw_pred = logits.argmax(1).cpu().numpy()
        abstain += (~IS_LEAF_FLAT[raw_pred]).sum()
        leaf_logits = logits[:, FLAT_OFFSET_LEAF:]
        pred_leaf = leaf_logits.argmax(1).cpu().numpy()
        true_leaf_g = val_leaf_local[i0:i0 + 2048]
        correct += (pred_leaf == true_leaf_g).sum()
    return 100.0 * correct / len(Xva_t), 100.0 * abstain / len(Xva_t)

# ============================================================
# 6. Bucle principal: 3 semillas x {flat, unmasked, masked}
# ============================================================
if QUICK_TEST:
    SEEDS = [1]
    N_ROUNDS = 2
    EVAL_EVERY = 1

results = {}
for seed in SEEDS:
    print(f"\n========== SEMILLA {seed} ==========")
    client_of_sample, depth_of_sample = make_partition(seed)
    np.savez(os.path.join(CKPT_DIR, f"partition_seed{seed}.npz"),
            client_of_sample=client_of_sample, depth_of_sample=depth_of_sample)

    Xtr_t = torch.tensor(Xtr_full, dtype=torch.float32)

    print(f"-- flat --")
    flat_model = train_flat(seed, basic_local, subord_local, leaf_local, depth_of_sample,
                            client_of_sample, Xtr_t)
    flat_acc, flat_abstain = eval_flat(flat_model)
    print(f"  flat: leaf_acc={flat_acc:.2f}%  abstencion={flat_abstain:.2f}%")

    print(f"-- unmasked --")
    _, log_un = train_tree(False, seed, basic_local, subord_local, leaf_local, depth_of_sample,
                           client_of_sample, Xtr_t, "unmasked")

    print(f"-- masked --")
    _, log_mk = train_tree(True, seed, basic_local, subord_local, leaf_local, depth_of_sample,
                           client_of_sample, Xtr_t, "masked")

    results[seed] = {
        "flat_leaf_acc": flat_acc, "flat_abstain": flat_abstain,
        "unmasked_leaf_acc": log_un["leaf_acc"][-1],
        "masked_leaf_acc": log_mk["leaf_acc"][-1],
    }
    print(f"seed {seed} resumen: {results[seed]}")

json.dump(results, open(os.path.join(CKPT_DIR, "imagenet3lf_main_results.json"), "w"))

print("\n=== RESUMEN FINAL (mean +/- std sobre semillas) ===")
for key in ["flat_leaf_acc", "flat_abstain", "unmasked_leaf_acc", "masked_leaf_acc"]:
    vals = [results[s][key] for s in SEEDS]
    print(f"{key}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")
