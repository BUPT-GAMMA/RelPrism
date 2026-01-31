from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.nn import HeteroConv, LayerNorm, TransformerConv


class HeteroGraphSAGEWithEdgeAttr(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
        heads: int = 4,
        edge_types_with_attr: Optional[List[EdgeType]] = None,
    ):
        super().__init__()
        
        self.num_heads = heads
        self.channels = channels
        self.edge_types_with_attr = set(edge_types_with_attr) if edge_types_with_attr is not None else set()
        
        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                out_channels = channels
                concat = False
            else:
                out_channels = channels
                concat = True
            
            conv = HeteroConv(
                {
                    edge_type: TransformerConv(
                        in_channels=channels if layer_idx == 0 else self.num_heads * channels,
                        out_channels=out_channels,
                        heads=self.num_heads,
                        concat=concat,
                        edge_dim=channels if edge_type in self.edge_types_with_attr else None,
                        aggr=aggr,
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            if layer_idx == num_layers - 1:
                norm_dim = channels
            else:  
                norm_dim = self.num_heads * channels
            
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(norm_dim, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
    ) -> Dict[NodeType, Tensor]:
        if edge_attr_dict is None:
            edge_attr_dict = {}
        else:
            edge_attr_dict = {
                edge_type: edge_attr 
                for edge_type, edge_attr in edge_attr_dict.items() 
                if edge_attr is not None and edge_type in self.edge_types_with_attr
            }
        
        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict_new = conv(x_dict, edge_index_dict, edge_attr_dict)
            
            if layer_idx == len(self.convs) - 1:
                expected_dim = self.channels
            else:
                expected_dim = self.num_heads * self.channels
            
            for node_type in x_dict.keys():
                if node_type not in x_dict_new or x_dict_new[node_type] is None:
                    input_feat = x_dict[node_type]
                    input_dim = input_feat.size(-1)
                    if input_dim != expected_dim:
                        repeat_times = (expected_dim + input_dim - 1) // input_dim  # 向上取整
                        expanded_feat = input_feat.repeat(1, repeat_times)
                        x_dict_new[node_type] = expanded_feat[:, :expected_dim]
                    else:
                        x_dict_new[node_type] = input_feat
            
            x_dict = x_dict_new
            x_dict = {key: x for key, x in x_dict.items() if x is not None}
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        num_layers: int,
        channels: int,
        heads: int,
        out_channels: int,
        aggr: str,
        norm: str,
    ):
        super().__init__()
        
        self.channels = channels
        
        self.node_feature_mapping = nn.ModuleDict()
        for node_type in data.node_types:
            self.node_feature_mapping[node_type] = nn.Linear(data[node_type].feat.shape[1], channels)
        self.edge_feature_mapping = nn.ModuleDict()
        edge_types_with_attr = []
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], "feat"):
                self.edge_feature_mapping[str(edge_type)] = nn.Linear(data[edge_type].feat.shape[1], channels)
                edge_types_with_attr.append(edge_type)

        # encoder
        self.gnn = HeteroGraphSAGEWithEdgeAttr(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            heads=heads,
            aggr=aggr,
            num_layers=num_layers,
            edge_types_with_attr=edge_types_with_attr,
        )

        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
    
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for node_feature_mapping in self.node_feature_mapping.values():
            node_feature_mapping.reset_parameters()
        for edge_feature_mapping in self.edge_feature_mapping.values():
            edge_feature_mapping.reset_parameters()
      
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        use_head: bool = False,
    ) -> Tensor:

        x_dict = {node_type: self.node_feature_mapping[node_type](batch[node_type].feat.float()) for node_type in batch.node_types}
        edge_attr_dict = {}
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "feat") and str(edge_type) in self.edge_feature_mapping:
                edge_attr_dict[edge_type] = self.edge_feature_mapping[str(edge_type)](batch[edge_type].feat.float())

        x_dict = self.gnn(x_dict, batch.edge_index_dict, edge_attr_dict)
        target_node_x = x_dict[entity_table]

        return self.head(target_node_x)

    
    def encode(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        x_dict = {node_type: self.node_feature_mapping[node_type](batch[node_type].feat.float()) for node_type in batch.node_types}
        edge_attr_dict = {}
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "feat") and str(edge_type) in self.edge_feature_mapping:
                edge_attr_dict[edge_type] = self.edge_feature_mapping[str(edge_type)](batch[edge_type].feat.float())
        x_dict = self.gnn(x_dict, batch.edge_index_dict, edge_attr_dict)
        return x_dict[entity_table]
        