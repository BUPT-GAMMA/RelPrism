import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["SCIPY_THREADS"] = "2"

import gc
import random
import argparse
from pathlib import Path
from sklearn.decomposition import PCA

import torch
torch.set_num_threads(1)
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from transformer_model import Model  
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything


from relbench.base import Dataset, EntityTask
from relbench.datasets import get_dataset
from custom_graph import make_graph_without_fact_tables
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import scatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--target_node_type", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for GNN backbone")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="last")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("/local/yjy/relbench_examples"))
parser.add_argument("--k_shot", type=int, default=5, help="Number of support samples per class")
parser.add_argument("--n_way", type=int, default=2, help="Number of clusters/classes per episode")
parser.add_argument("--meta_batch_size", type=int, default=32, help="Episodes per epoch")
parser.add_argument("--max_clusters", type=int, default=20, help="Number of clusters/classes per episode")
parser.add_argument("--accum_steps", type=int, default=16, help="Gradient accumulation steps")
parser.add_argument("--use_structure", type=int, default=1, help="Whether to use structure features")
parser.add_argument("--is_eval", type=int, default=1, help="1: eval, 0: train")
args = parser.parse_args()

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Data Loading ---
dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

args.target_node_type = task.entity_table

from utils import get_configs
model_config, loader_config = get_configs(args.dataset, args.task)


model_config['num_model_layers'] = int(model_config['num_model_layers'] / 2) if model_config['num_model_layers'] > 1 else 1

fact_tables, data, all_node_time = make_graph_without_fact_tables(
    db=dataset.get_db(),
    feat_data_dir=f"./data/{args.dataset}-final",
    dataset_name=args.dataset,
)

def inject_statistical_features(data, target_node_type, structural_weight=10.0):
    
    if target_node_type in all_node_time:
        node_cutoff_times = all_node_time[target_node_type].cpu()
    else:
        node_cutoff_times = torch.zeros(data[target_node_type].num_nodes, dtype=torch.long)

    raw_feat = data[target_node_type].feat.cpu()
    num_nodes = data[target_node_type].num_nodes
    
    data[target_node_type].feat = None 
    gc.collect()
    
    new_feats_list = []
    
    def get_valid_edges(edge_type, mode='src'):
        edge_index = data[edge_type].edge_index.cpu()
        
        if hasattr(data[edge_type], 'time') and data[edge_type].time is not None:
            edge_time = data[edge_type].time.cpu()
            
            target_indices = edge_index[0] if mode == 'src' else edge_index[1]
            
            cutoff = node_cutoff_times[target_indices]
            
            mask = (edge_time <= cutoff) | (cutoff == 0)
            
            valid_edge_index = edge_index[:, mask]
            
            valid_edge_attr = None
            if hasattr(data[edge_type], 'feat') and data[edge_type].feat is not None:
                valid_edge_attr = data[edge_type].feat.cpu()[mask]
                
            return valid_edge_index, valid_edge_attr
        
        else:
            edge_attr = None
            if hasattr(data[edge_type], 'feat') and data[edge_type].feat is not None:
                edge_attr = data[edge_type].feat.cpu()
            return edge_index, edge_attr

    node_degree = torch.zeros(num_nodes)
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        
        if src == target_node_type:
            valid_edge_index, _ = get_valid_edges(edge_type, mode='src')
            if valid_edge_index.size(1) > 0:
                d = pyg_utils.degree(valid_edge_index[0], num_nodes=num_nodes)
                node_degree += d
                
        elif dst == target_node_type:
            valid_edge_index, _ = get_valid_edges(edge_type, mode='dst')
            if valid_edge_index.size(1) > 0:
                d = pyg_utils.degree(valid_edge_index[1], num_nodes=num_nodes)
                node_degree += d
                
    new_feats_list.append(torch.log1p(node_degree).unsqueeze(1))

    edge_attr_mean = torch.zeros(num_nodes)
    edge_attr_sum = torch.zeros(num_nodes)
    has_edge_attr = False
    
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        if src == target_node_type:
            valid_edge_index, valid_e_attr = get_valid_edges(edge_type, mode='src')
            
            if valid_e_attr is not None:
                has_edge_attr = True
                if valid_e_attr.dim() > 1 and valid_e_attr.shape[1] > 1:
                    e_val = torch.norm(valid_e_attr, p=2, dim=1)
                else:
                    e_val = valid_e_attr.float().squeeze()
                
                try:
                    agg_mean = scatter(e_val, valid_edge_index[0], dim=0, dim_size=num_nodes, reduce='mean')
                    edge_attr_mean += agg_mean
                    agg_sum = scatter(e_val, valid_edge_index[0], dim=0, dim_size=num_nodes, reduce='sum')
                    edge_attr_sum += agg_sum
                except Exception:
                    pass
    
    if has_edge_attr:
        new_feats_list.append(edge_attr_mean.unsqueeze(1))
        new_feats_list.append(torch.log1p(edge_attr_sum).unsqueeze(1))
    
    neighbor_norm_agg = torch.zeros(num_nodes)
    for edge_type in data.edge_types:
        src, _, dst = edge_type
        if src == target_node_type and dst != target_node_type:
            valid_edge_index, _ = get_valid_edges(edge_type, mode='src')
            
            if hasattr(data[dst], 'feat') and valid_edge_index.size(1) > 0:
                dst_feat = data[dst].feat.cpu()
                dst_norm = torch.norm(dst_feat, p=2, dim=1)
                
                try:
                    agg = scatter(dst_norm[valid_edge_index[1]], valid_edge_index[0], dim=0, dim_size=num_nodes, reduce='mean')
                    neighbor_norm_agg += agg
                except: pass
                
    new_feats_list.append(torch.log1p(neighbor_norm_agg).unsqueeze(1))
    
    self_norm = torch.norm(raw_feat, p=2, dim=1).unsqueeze(1)
    new_feats_list.append(torch.log1p(self_norm))
    
    structural_feat = torch.cat(new_feats_list, dim=1)

    del new_feats_list
    gc.collect()

    scaler = StandardScaler()
    structural_feat_np = structural_feat.numpy()
    structural_feat_np = scaler.fit_transform(structural_feat_np)
    structural_feat = torch.from_numpy(structural_feat_np).float()
    del structural_feat_np
    
    structural_feat.mul_(structural_weight)

    combined_feat = torch.cat([raw_feat, structural_feat], dim=1)
    added_dim = structural_feat.shape[1]

    del raw_feat
    del structural_feat
    gc.collect()
    
    data[target_node_type].feat = combined_feat.cpu()

    return data, added_dim

target_added_dim = 0
if args.use_structure == 1:
    for node_type in data.node_types:
        current_weight = 10.0 if node_type == args.target_node_type else 5.0
        
        data, added_dim = inject_statistical_features(
            data, 
            target_node_type=node_type, 
            structural_weight=current_weight
        )
        if node_type == args.target_node_type:
            target_added_dim = added_dim


def create_loader(graph, input_time, node_indices, batch_size=None):
    fanouts = [int(loader_config['num_neighbors'] / 2**i) for i in range(model_config['num_model_layers'])]
    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)
    if not isinstance(input_time, torch.Tensor):
        input_time = torch.tensor(input_time, dtype=torch.long)
    
    final_bs = batch_size if batch_size is not None else args.batch_size
    final_bs = min(final_bs, len(node_indices))

    if not torch.all(input_time == 0):
        loader = NeighborLoader(
            graph,
            num_neighbors=fanouts,
            time_attr="time",
            input_nodes=(args.target_node_type, node_indices),
            input_time=input_time,
            batch_size=final_bs,
            temporal_strategy=args.temporal_strategy,
            shuffle=False,
            subgraph_type=loader_config['subgraph_type'],
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    else:
        loader = NeighborLoader(
            graph,
            num_neighbors=fanouts,
            input_nodes=(args.target_node_type, node_indices),
            batch_size=final_bs,
            shuffle=False,
            disjoint=True,
            subgraph_type=loader_config['subgraph_type'],
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    return loader

def load_split(path, split, col_type):
    root = Path(path)
    npz_path = root / f"{split}.npz"
    if not npz_path.exists(): raise FileNotFoundError(f"File not found: {npz_path}")
    data_npz = np.load(npz_path)
    node_ids, times, labels = None, None, None
    for col in data_npz.keys():
        if col in col_type:
            if col_type[col] == "primary_key": node_ids = data_npz[col]
            elif col_type[col] == "timestamp": times = data_npz[col]
            elif col_type[col] == "label": labels = data_npz[col]
    return node_ids, times, labels

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1): raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def compute_prototypes(support_embeddings, support_labels, n_way):
    dim = support_embeddings.size(1)
    prototypes = torch.zeros([n_way, dim], device=support_embeddings.device)
    for c in range(n_way):
        mask = (support_labels == c)
        if mask.sum() > 0:
            prototypes[c] = support_embeddings[mask].mean(0)
    return prototypes

def pre_generate_all_tasks(total_tasks, pca_dim=128, max_clusters=args.max_clusters, num_clustering_views=100, structure_dim=5):
    num_struct_feats = structure_dim
    full_x = data[args.target_node_type].feat.cpu().numpy()
    all_node_ids = np.arange(full_x.shape[0])

    feat_structural_norm = None
    
    if structure_dim > 0:
        num_struct_feats = structure_dim
        feat_semantic = full_x[:, :-num_struct_feats].copy()
        feat_structural = full_x[:, -num_struct_feats:].copy()
        
        scaler = StandardScaler()
        feat_structural_norm = scaler.fit_transform(feat_structural)
        del scaler

        del full_x, feat_structural
        gc.collect()

    else:
        feat_semantic = full_x
        del full_x
    
    effective_dim = min(pca_dim, feat_semantic.shape[1])
    if feat_semantic.shape[0] > 50000:
        indices = np.random.choice(feat_semantic.shape[0], 50000, replace=False)
        pca = PCA(n_components=effective_dim, random_state=42)
        pca.fit(feat_semantic[indices])
        x_semantic_pca = pca.transform(feat_semantic)
    else:
        pca = PCA(n_components=effective_dim, random_state=42)
        x_semantic_pca = pca.fit_transform(feat_semantic)
    
    del feat_semantic
    gc.collect()
    
    pools = {
        "structural": [],
        "semantic": [],
        "hybrid": []
    }

   
    if feat_structural_norm is not None:
        x_hybrid = np.concatenate([x_semantic_pca, feat_structural_norm * 5.0], axis=1)
        
        n_struct = int(num_clustering_views * 0.3)
        n_hybrid = int(num_clustering_views * 0.5)
        n_semantic = num_clustering_views - n_struct - n_hybrid
        
        view_configs = [
            ("structural", feat_structural_norm, n_struct),
            ("hybrid", x_hybrid, n_hybrid),
            ("semantic", x_semantic_pca, n_semantic)
        ]
        pool_choices = ["structural", "hybrid", "semantic"]
        pool_weights = [0.3, 0.5, 0.2] 
        
    else:
        
        n_semantic = num_clustering_views
        view_configs = [
            ("semantic", x_semantic_pca, n_semantic)
        ]
        pool_choices = ["semantic"]
        pool_weights = [1.0]
    
    for pool_name, feat_data, n_views in view_configs:
        if n_views <= 0: continue
        
        for _ in tqdm(range(n_views), desc=f"Building {pool_name} pool", leave=False):
            k_current = random.randint(args.n_way, max_clusters)
            
            kmeans = MiniBatchKMeans(
                n_clusters=k_current, 
                batch_size=4096,
                n_init='auto',
                random_state=random.randint(0, 100000)
            )
            cluster_labels = kmeans.fit_predict(feat_data)
            
            unique_cls, counts = np.unique(cluster_labels, return_counts=True)
            valid_cls = unique_cls[counts >= 2 * args.k_shot]
            
            if len(valid_cls) >= args.n_way:
                pools[pool_name].append({"labels": cluster_labels, "valid_cls": valid_cls})

    task_queue = []
    for _ in range(total_tasks):
        chosen_pool_name = random.choices(pool_choices, weights=pool_weights, k=1)[0]
        
        if not pools[chosen_pool_name]:
            available_pools = [k for k, v in pools.items() if len(v) > 0]
            chosen_pool_name = random.choice(available_pools)
            
        view = random.choice(pools[chosen_pool_name])
        
        cluster_labels = view["labels"]
        valid_cls = view["valid_cls"]
        
        selected_cls = np.random.choice(valid_cls, args.n_way, replace=False)
        cls_map = {c: i for i, c in enumerate(selected_cls)}
        
        support_nids, query_nids = [], []
        support_lbls, query_lbls = [], []
        
        for c in selected_cls:
            indices = all_node_ids[cluster_labels == c]
            if len(indices) > 2 * args.k_shot:
                chosen = np.random.choice(indices, 2 * args.k_shot, replace=False)
            else:
                chosen = indices
                
            support_nids.extend(chosen[:args.k_shot])
            support_lbls.extend([cls_map[c]] * len(chosen[:args.k_shot]))
            
            query_nids.extend(chosen[args.k_shot:])
            query_lbls.extend([cls_map[c]] * len(chosen[args.k_shot:]))
            
        task_queue.append((
            torch.tensor(support_nids, dtype=torch.long),
            torch.tensor(query_nids, dtype=torch.long),
            torch.tensor(support_lbls, dtype=torch.long).to(device),
            torch.tensor(query_lbls, dtype=torch.long).to(device)
        ))
        
    print(f"Task Generated...")
    return task_queue

def run_experiment(seed: int):
    print(f"\n{'='*60}\nRun Seed: {seed}\n{'='*60}")
    seed_everything(seed)

    torch.cuda.empty_cache()
    
    model = Model(
        data=data,
        num_layers=model_config['num_model_layers'],
        channels=model_config['channels'],
        heads=model_config['num_heads'],
        out_channels=1, 
        aggr=args.aggr,
        norm="batch_norm",
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    griffin_root = f"./data/{args.dataset}-final"
    import yaml
    metadata_path = Path(griffin_root) / "metadata.yaml"
    with open(metadata_path, "r") as f: metadata = yaml.safe_load(f)
    task_info = next((t for t in metadata.get("tasks", []) if t.get("name") == args.task), None)
    col_type = {}
    for col in task_info.get("columns", []):
        if col.get("dtype") == "primary_key": col_type[col.get("name")] = "primary_key"
        elif col.get("dtype") == "timestamp": col_type[col.get("name")] = "timestamp"
        elif col.get("is_target_column"): col_type[col.get("name")] = "label"

    task_data_dir = Path(griffin_root) / args.task
    train_ids, train_times, train_labels = load_split(task_data_dir, split="train", col_type=col_type)
    test_ids, test_times, test_labels = load_split(task_data_dir, split="test", col_type=col_type)

    node_to_label = {int(nid): int(lbl) for nid, lbl in zip(train_ids.tolist(), train_labels.tolist())}
    label_to_nodes = {}
    for nid, lbl in node_to_label.items(): label_to_nodes.setdefault(lbl, []).append(nid)
    
    downstream_support_ids = []
    for c in list(label_to_nodes.keys()):
        nodes = label_to_nodes[c]
        sampled = random.sample(nodes, args.k_shot) if len(nodes) >= args.k_shot else nodes
        downstream_support_ids.extend(sampled)
    
    id_to_pos = {int(nid): i for i, nid in enumerate(train_ids.tolist())}
    support_positions = [id_to_pos[nid] for nid in downstream_support_ids]
    downstream_support_times = torch.tensor(train_times[support_positions], dtype=torch.long)
    downstream_support_ids_t = torch.tensor(downstream_support_ids, dtype=torch.long)
    downstream_support_labels = torch.tensor(train_labels[support_positions], dtype=torch.long).to(device)

    downstream_support_loader = create_loader(data, downstream_support_times, downstream_support_ids_t, batch_size=32)
    test_loader = create_loader(data, torch.tensor(test_times, dtype=torch.long), torch.tensor(test_ids, dtype=torch.long), batch_size=32)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long).to(device)

    def get_embeddings(loader):
        model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)

                if hasattr(batch[args.target_node_type], 'batch_size'):
                    current_batch_size = batch[args.target_node_type].batch_size
                else:
                    current_batch_size = batch[args.target_node_type].num_nodes
                
                full_emb = model.encode(batch, args.target_node_type)
                
                embeddings.append(full_emb[:current_batch_size].cpu())

        return torch.cat(embeddings, dim=0)
    
    model_path = f"./pretrained_model/{args.dataset}_{args.task}_pretrained_{args.k_shot}_42.pt"
    
    if args.is_eval == 0 and seed == 42:
        total_required_tasks = args.epochs * args.meta_batch_size
        buffer_size = int(total_required_tasks * 1.1)
        
        task_queue = pre_generate_all_tasks(buffer_size, pca_dim=128, structure_dim=target_added_dim)
        
        task_iterator = iter(task_queue)
        
        print("\nStart Pre-training...")
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            pbar = tqdm(range(args.meta_batch_size), desc=f"Epoch {epoch+1}", leave=False)
            
            for i in pbar:
               
                try:
                    task_data = next(task_iterator)
                except StopIteration:
                    task_iterator = iter(task_queue)
                    task_data = next(task_iterator)
                
                s_nids, q_nids, s_lbls, q_lbls = task_data
                
                s_times = all_node_time[args.target_node_type][s_nids]
                q_times = all_node_time[args.target_node_type][q_nids]
                
                s_loader = create_loader(data, s_times, s_nids)
                q_loader = create_loader(data, q_times, q_nids)
                
                s_emb_list = []
                for batch in s_loader:
                    batch = batch.to(device)
                    current_batch_size = batch[args.target_node_type].batch_size
                    
                    full_emb = model.encode(batch, args.target_node_type)
                    s_emb_list.append(full_emb[:current_batch_size])
                    
                    del full_emb, batch
                s_emb = torch.cat(s_emb_list, dim=0)
                del s_emb_list
                torch.cuda.empty_cache()
                
                
                q_emb_list = []
                for batch in q_loader:
                    batch = batch.to(device)
                    current_batch_size = batch[args.target_node_type].batch_size
                    full_emb = model.encode(batch, args.target_node_type)
                    q_emb_list.append(full_emb[:current_batch_size])
                    del full_emb, batch
                q_emb = torch.cat(q_emb_list, dim=0)
                del q_emb_list
                torch.cuda.empty_cache()
                
                current_n_way = len(torch.unique(s_lbls))
                prototypes = compute_prototypes(s_emb, s_lbls, current_n_way)
                dists = euclidean_dist(q_emb, prototypes)
                log_p_y = F.log_softmax(-dists, dim=1)
                loss = F.nll_loss(log_p_y, q_lbls)
                
                loss = loss / args.accum_steps
                loss.backward()
                
                total_loss += loss.item() * args.accum_steps
                
                del prototypes, dists, log_p_y, loss
                
                if ((i + 1) % args.accum_steps == 0) or ((i + 1) == args.meta_batch_size):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    del s_emb, q_emb
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / args.meta_batch_size
            print(f"Epoch {epoch+1:02d} | ProtoLoss: {avg_loss:.6f}")
        print(f"Model Saved to {model_path}")
        torch.save(model.state_dict(), model_path)
    
    else:
        print(f"Loading Pretrained Model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    torch.cuda.empty_cache()
    print("\nEvaluating...")
    model.eval()
    support_embs = get_embeddings(downstream_support_loader)
    downstream_support_labels_cpu = downstream_support_labels.cpu()
    downstream_prototypes = compute_prototypes(support_embs, downstream_support_labels_cpu, 2)
    test_embs = get_embeddings(test_loader)
    dists = euclidean_dist(test_embs, downstream_prototypes)
    probs = F.softmax(-dists, dim=1)
    pos_probs = probs[:, 1].cpu().numpy()
    auc = roc_auc_score(test_labels_t.cpu().numpy(), pos_probs)
    print(f"Test ROC-AUC: {auc:.6f}")
    return auc

num_runs = 5
seeds = [args.seed + i for i in range(num_runs)]
all_results = []

for i, seed in enumerate(seeds):
    print(f"\nRun {i+1}/{num_runs}")
    res = run_experiment(seed)
    all_results.append(res)

print(f"\nFinal Results: {np.mean(all_results):.4f} ± {np.std(all_results):.4f}")