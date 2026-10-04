import os
import io
from PIL import Image

def generate_sample_arrows(save_dir="data/finetune_arrows"):
    try:
        import pandas as pd
        import pyarrow as pa
    except ImportError:
        print("⚠️ pandas or pyarrow not installed yet.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    needed = ["vqa_vqa_rad_train.arrow", "vqa_vqa_rad_val.arrow", "vqa_vqa_rad_test.arrow"]
    if all(os.path.exists(os.path.join(save_dir, f)) for f in needed):
        print(f"✅ VQA-RAD arrow dataset files present in '{save_dir}'")
        return

    print("⚡ Generating sample VQA-RAD arrow files for testing & validation...")
    
    # Create sample synthetic medical image (grey scale simulation)
    img = Image.new('RGB', (384, 384), color=(128, 128, 128))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    for split in ['train', 'val', 'test']:
        num_samples = 20 if split == 'train' else 5
        bs = []
        for i in range(num_samples):
            bs.append([
                img_bytes,
                [f"Is there abnormality in lung region {i}?"],
                [["yes"]],
                [[0]],
                [[1.0]],
                f"sample_rad_{split}_{i}.jpg",
                [i],
                [0],  # 0=closed, 1=open
                split
            ])
        df = pd.DataFrame(
            bs,
            columns=[
                'image', 'questions', 'answers', 'answer_labels', 'answer_scores',
                'image_id', 'question_id', 'answer_type', 'split'
            ]
        )
        table = pa.Table.from_pandas(df)
        arrow_path = os.path.join(save_dir, f"vqa_vqa_rad_{split}.arrow")
        with pa.OSFile(arrow_path, 'wb') as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print(f"  --> Created {arrow_path} ({num_samples} samples)")

def generate_external_graph_feats(ext_dir="data/external_data"):
    try:
        import torch
        os.makedirs(ext_dir, exist_ok=True)
        adj_path = os.path.join(ext_dir, "adj_matrix.pt")
        organ_path = os.path.join(ext_dir, "organ_disease_info.pt")
        edge_index_path = os.path.join(ext_dir, "edge_index.pt")
        edge_type_path = os.path.join(ext_dir, "edge_type.pt")
        
        num_nodes = 577
        num_relations = 5  # 0: is_part_of, 1: manifests_as, 2: located_in, 3: adjacent_to, 4: self_loop

        edges = []
        edge_types = []

        # 1. Self-loops (Relation type 4)
        for i in range(num_nodes):
            edges.append((i, i))
            edge_types.append(4)

        # 2. Structured Anatomical & Disease Relational Triplets
        # 0: is_part_of (Sub-organ -> Organ -> Region)
        # 1: manifests_as (Disease -> Finding)
        # 2: located_in (Disease -> Organ)
        # 3: adjacent_to (Organ -> Organ)

        # Organs (nodes 0 to 49)
        # Sub-regions (nodes 50 to 149)
        # Diseases (nodes 150 to 349)
        # Findings (nodes 350 to 576)

        # Build part_of relations (50..149 -> 0..49)
        for sub in range(50, 150):
            parent_organ = (sub - 50) % 50
            edges.append((sub, parent_organ))
            edge_types.append(0)  # is_part_of
            edges.append((parent_organ, sub))
            edge_types.append(0)

        # Build located_in relations (150..349 -> 0..49)
        for dis in range(150, 350):
            target_organ = (dis - 150) % 50
            edges.append((dis, target_organ))
            edge_types.append(2)  # located_in
            edges.append((target_organ, dis))
            edge_types.append(2)

        # Build manifests_as relations (150..349 -> 350..576)
        for dis in range(150, 350):
            finding = 350 + ((dis - 150) % 227)
            edges.append((dis, finding))
            edge_types.append(1)  # manifests_as
            edges.append((finding, dis))
            edge_types.append(1)

        # Build adjacent_to relations among organs (0..49)
        for org in range(0, 49):
            adj_org = (org + 1) % 50
            edges.append((org, adj_org))
            edge_types.append(3)  # adjacent_to

        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous() # [2, E]
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long) # [E]

        # Dense adjacency matrix for backward compatibility
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for src, dst in edges:
            adj_matrix[src, dst] = 1.0

        torch.save(adj_matrix, adj_path)
        torch.save(torch.randint(0, 30522, (1, num_nodes), dtype=torch.long), organ_path)
        torch.save(edge_index_tensor, edge_index_path)
        torch.save(edge_type_tensor, edge_type_path)

        print(f"✅ Pre-generated Relational Knowledge Graph edge index tensor: {edge_index_path} (Edges: {edge_index_tensor.size(1)})")
        print(f"✅ Pre-generated Relational Knowledge Graph edge type tensor: {edge_type_path} (Relations: {num_relations})")
        print(f"✅ Pre-generated Knowledge Graph adjacency tensor: {adj_path}")
        print(f"✅ Pre-generated Knowledge Graph token tensor: {organ_path}")
    except Exception as e:
        print(f"⚠️ Could not generate graph features: {e}")

if __name__ == "__main__":
    generate_external_graph_feats()
    generate_sample_arrows()
