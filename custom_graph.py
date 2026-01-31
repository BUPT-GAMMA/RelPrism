from __future__ import annotations

from typing import Any, Dict, Tuple, List

import os
import time
import numpy as np
import torch
from torch_geometric.data import HeteroData

from relbench.base import Database
from torch_frame import stype as tf_stype
from torch_frame.data.stats import StatType


def _classify_tables(db: Database) -> Tuple[List[str], List[str], List[str]]:
    referenced_tables = set()
    for table in db.table_dict.values():
        for pkey_table in table.fkey_col_to_pkey_table.values():
            referenced_tables.add(pkey_table)
    
    fact_tables = []
    node_tables_single_fkey = []
    node_tables_multi_fkey = []
    
    for table_name, table in db.table_dict.items():
        num_fkeys = len(table.fkey_col_to_pkey_table)
        is_referenced = table_name in referenced_tables
        
        if num_fkeys >= 2 and not is_referenced:
            fact_tables.append(table_name)
        elif num_fkeys <= 1:
            node_tables_single_fkey.append(table_name)
        elif num_fkeys >= 2 and is_referenced:
            node_tables_multi_fkey.append(table_name)
    
    return fact_tables, node_tables_single_fkey, node_tables_multi_fkey


def _aggregate_fact_rows(table, table_name, src_table, dst_table, src_col, dst_col):
    
    start_time = time.time()
    
    if table_name is not None and src_table is not None and dst_table is not None:
        print(f"Building edge: {src_table} -> {dst_table} (fact_table={table_name}, fkey_pair=({src_col}, {dst_col})")

    src = table.foreign_key[src_col]
    dst = table.foreign_key[dst_col]
    num_edges = len(src)

    edge_time = table.time
    edge_feat = table.feat

    valid_mask = (src >= 0) & (dst >= 0)
    src = src[valid_mask]
    dst = dst[valid_mask]
    
    if edge_time is not None:
        edge_time = edge_time[valid_mask]
    if edge_feat is not None:
        edge_feat = edge_feat[valid_mask]

    edge_index = torch.stack([src, dst], dim=0)

    
    elapsed_time = time.time() - start_time
    print(f"_aggregate_fact_rows time: {elapsed_time:.2f} seconds")

    return edge_index, edge_time, edge_feat

from pathlib import Path
import yaml
def read_dtype_from_metadata(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    with open(metadata_path, "r") as f:
        meta = yaml.safe_load(f)

    table_to_col_dtype: Dict[str, Dict[str, str]] = {}
    tables = meta.get("tables", [])
    for table in tables:
        tname = table.get("name")
        if not tname:
            print(f"Table name not found: {metadata_path}")
            continue
        col_dtype_dict = {}
        for col in table.get("columns", []):
            cname = col.get("name")
            if not cname:
                print(f"Column name not found: {metadata_path}")
                continue

            dtype = col.get("dtype")
            if dtype:
                col_dtype_dict[cname] = dtype
        if col_dtype_dict:
            table_to_col_dtype[tname] = col_dtype_dict
    return table_to_col_dtype


def make_graph_without_fact_tables(
    db: Database,
    feat_data_dir: str,
    min_pair_count: int = 1,
    dataset_name: str = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]], Dict[Tuple[str, str, str], Dict[tf_stype, List[str]]]]:
    data = HeteroData()

    fact_tables, node_tables_single_fkey, node_tables_multi_fkey = _classify_tables(db)
    
    metadata_path = feat_data_dir + "/metadata.yaml"
    metadata_path = Path(metadata_path).resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    table_to_col_dtype = read_dtype_from_metadata(metadata_path)

    for table_name, table in db.table_dict.items():
        path = os.path.join(feat_data_dir, table_name + ".npz")
        try:
            feat = np.load(path)
        except Exception as e:
            print(f"\n[ERROR] Failed to load {table_name} from {path}: {e}")
            continue

        print(f"\n===== Table: {table_name} =====")
        feat_list = []
        data[table_name].foreign_key = {}
        for col_name in feat.files:
            arr = feat[col_name]
            dtype = table_to_col_dtype[table_name][col_name]
            if dtype == "primary_key":
                data[table_name].primary_key = torch.from_numpy(arr)
            elif dtype == "foreign_key":
                data[table_name].foreign_key[col_name] = torch.from_numpy(arr)
            elif dtype == "timestamp":
                data[table_name].time = torch.from_numpy(arr)
            else:
                feat_list.append(arr)
        feat_concat = np.concatenate(feat_list, axis=1)
        data[table_name].feat = torch.from_numpy(feat_concat)

    for table_name in fact_tables:
        table = db.table_dict[table_name]

        fkeys = list(table.fkey_col_to_pkey_table.items())

        time_col = table.time_col

        valid_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
        df_fk = table.df
        for i in range(len(fkeys)):
            for j in range(i + 1, len(fkeys)):
                (fkey_a, ptable_a), (fkey_b, ptable_b) = fkeys[i], fkeys[j]
                mask = (~df_fk[fkey_a].isna()) & (~df_fk[fkey_b].isna())
                if time_col is not None:
                    mask = mask & (~df_fk[time_col].isna())
                if int(mask.sum()) >= min_pair_count:
                    valid_pairs.append(((fkey_a, ptable_a), (fkey_b, ptable_b)))

        for (fkey_a, ptable_a), (fkey_b, ptable_b) in valid_pairs:
            src_table = ptable_a
            dst_table = ptable_b

            print("keys: ", list(data[table_name].keys()))

            edge_index, edge_time, edge_feat = _aggregate_fact_rows(
                table=data[table_name],
                table_name=table_name,
                src_table=src_table,
                dst_table=dst_table,
                src_col=fkey_a,
                dst_col=fkey_b,
            )

            is_self_loop = (src_table == dst_table)
            
            if is_self_loop:
                edge_type = (src_table, f"self_fact_{table_name}_{fkey_a}_to_{fkey_b}", dst_table)
                print(f"Self-loop edge detected: {edge_type} (fkey_pair=({fkey_a}, {fkey_b})")
            else:
                edge_type = (src_table, f"fact_{table_name}_{fkey_a}_{fkey_b}", dst_table)
            
            data[edge_type].edge_index = edge_index
            data[edge_type].time = edge_time
            data[edge_type].feat = edge_feat

            if not is_self_loop:
                rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                rev_edge_type = (dst_table, f"rev_fact_{table_name}_{fkey_a}_{fkey_b}", src_table)
    
                data[rev_edge_type].edge_index = rev_edge_index
                data[rev_edge_type].time = edge_time
                data[rev_edge_type].feat = edge_feat

    for table_name in node_tables_single_fkey:
        table = db.table_dict[table_name]
        
        if len(table.fkey_col_to_pkey_table) != 1:
            continue
        fkey_col, pkey_table = next(iter(table.fkey_col_to_pkey_table.items()))

        src_index = data[table_name].primary_key
        dst_index = data[table_name].foreign_key[fkey_col]

        valid_mask = (src_index >= 0) & (dst_index >= 0)
        src_index = src_index[valid_mask]
        dst_index = dst_index[valid_mask]

        edge_index = torch.stack([src_index, dst_index], dim=0)
        edge_time = data[table_name].time
        if edge_time is not None:
            edge_time = edge_time[valid_mask]
        edge_feat = None
        
        is_self_loop = (table_name == pkey_table)
        
        if is_self_loop:
            edge_type = (table_name, f"self_fkey_{table_name}_{fkey_col}", pkey_table)
            print(f"Self-loop edge detected: {edge_type} (fkey_column={fkey_col})")
        else:
            edge_type = (table_name, f"fkey_{fkey_col}", pkey_table)
        
        data[edge_type].edge_index = edge_index
        data[edge_type].time = edge_time
        
        data[edge_type].feat = edge_feat
        
        if not is_self_loop:
            rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            rev_edge_type = (pkey_table, f"rev_fkey_{fkey_col}", table_name)
            data[rev_edge_type].edge_index = rev_edge_index
           
            data[rev_edge_type].time = edge_time
            data[rev_edge_type].feat = edge_feat

    for table_name in node_tables_multi_fkey:
        table = db.table_dict[table_name]

        fkeys = list(table.fkey_col_to_pkey_table.items())  # >= 2

        for fkey_col, pkey_table in fkeys:
            src_index = data[table_name].primary_key
            dst_index = data[table_name].foreign_key[fkey_col]

            valid_mask = (src_index >= 0) & (dst_index >= 0)
            src_index = src_index[valid_mask]
            dst_index = dst_index[valid_mask]

            edge_index = torch.stack([src_index, dst_index], dim=0)
            edge_time = data[table_name].time
            if edge_time is not None:
                edge_time = edge_time[valid_mask]
            edge_feat = None
            
            is_self_loop = (table_name == pkey_table)
            
            if is_self_loop:
                edge_type = (table_name, f"self_fkey_{table_name}_{fkey_col}", pkey_table)
                print(f"Self-loop edge detected: {edge_type} (fkey_column={fkey_col})")
            else:
                edge_type = (table_name, f"fkey_{fkey_col}", pkey_table)
                print(f"Creating edge: {edge_type} (fkey_column={fkey_col}, target_table={pkey_table}, edge_count={edge_index.size(1)})")
            
            data[edge_type].edge_index = edge_index
            data[edge_type].time = edge_time
            data[edge_type].feat = edge_feat
            
            if not is_self_loop:
                rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                rev_edge_type = (pkey_table, f"rev_fkey_{fkey_col}", table_name)
        
                data[rev_edge_type].edge_index = rev_edge_index
                data[rev_edge_type].time = edge_time
                data[rev_edge_type].feat = edge_feat

    data.validate()

    for table_name in fact_tables:
        if table_name in data.node_types:
            del data[table_name]

    all_node_time = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'time'):
            all_node_time[node_type] = data[node_type].time

    for node_type in data.node_types:
        if hasattr(data[node_type], 'time'):
            delattr(data[node_type], 'time')

    return fact_tables, data, all_node_time