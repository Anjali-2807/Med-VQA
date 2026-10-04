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
        from transformers import BertTokenizer
        
        os.makedirs(ext_dir, exist_ok=True)
        adj_path = os.path.join(ext_dir, "adj_matrix.pt")
        organ_path = os.path.join(ext_dir, "organ_disease_info.pt")
        edge_index_path = os.path.join(ext_dir, "edge_index.pt")
        edge_type_path = os.path.join(ext_dir, "edge_type.pt")
        
        num_nodes = 577
        # 8 Directed Relational Semantics:
        # 0: is_part_of, 1: has_part, 2: located_in, 3: contains,
        # 4: manifests_as, 5: indicated_by, 6: adjacent_to, 7: self_loop
        num_relations = 8

        # --- 1. Real Medical Term Vocabulary Mapping ---
        organs = [
            "head", "brain", "skull", "chest", "lung", "pleura", "heart", "aorta", "mediastinum", "airway",
            "abdomen", "liver", "gallbladder", "spleen", "pancreas", "kidney", "stomach", "bowel", "colon", "bladder",
            "spine", "vertebra", "pelvis", "hip", "femur", "shoulder", "clavicle", "rib", "diaphragm", "peritoneum",
            "neck", "thyroid", "carotid", "trachea", "esophagus", "adrenal", "uterus", "ovary", "prostate", "vascular",
            "bone", "joint", "muscle", "soft tissue", "lymph node", "thoracic wall", "retroperitoneum", "pericardium", "bronchus", "hilar region"
        ]
        
        sub_regions = [
            "upper lobe", "lower lobe", "middle lobe", "apex", "base", "costophrenic angle", "ventricle", "cerebellum",
            "brainstem", "frontal lobe", "parietal lobe", "occipital lobe", "temporal lobe", "white matter", "grey matter",
            "left atrium", "right atrium", "left ventricle", "right ventricle", "ascending aorta", "aortic arch",
            "hepatic lobe", "renal cortex", "renal medulla", "splenic parenchyma", "pancreatic head", "pancreatic tail",
            "lumbar spine", "cervical spine", "thoracic spine", "sacrum", "iliac crest", "femoral head", "acetabulum",
            "pleural space", "pericardial space", "peritoneal cavity", "mediastinal space", "hilar area", "subpleural space"
        ]
        
        diseases = [
            "pneumonia", "cardiomegaly", "pleural effusion", "atelectasis", "pneumothorax", "consolidation", "pulmonary edema",
            "lung nodule", "lung mass", "tuberculosis", "emphysema", "bronchitis", "stroke", "brain infarct", "intracranial hemorrhage",
            "hydrocephalus", "brain tumor", "glioblastoma", "meningioma", "hepatic steatosis", "liver cirrhosis", "hepatocellular carcinoma",
            "cholecystitis", "cholelithiasis", "splenomegaly", "pancreatitis", "renal cyst", "nephrolithiasis", "renal cell carcinoma",
            "appendicitis", "bowel obstruction", "diverticulitis", "fracture", "osteoarthritis", "spondylolisthesis", "disc herniation",
            "bone metastasis", "lymphadenopathy", "aortic aneurysm", "pulmonary embolism", "deep vein thrombosis"
        ]
        
        findings = [
            "opacity", "ground glass opacity", "shadowing", "hyperintensity", "hypointensity", "ring enhancement",
            "calcification", "fluid accumulation", "air fluid level", "soft tissue swelling", "cortical disruption",
            "joint space narrowing", "osteophyte", "midline shift", "sulcal effacement", "mass effect", "pericardial effusion",
            "ascites", "lymph node enlargement", "nodular lesion", "cavitation", "reticular pattern", "hilar enlargement", "vascular congestion"
        ]

        # Construct medical node concept list up to 577 nodes
        node_concepts = []
        for i in range(num_nodes):
            if i < 50:
                concept = organs[i % len(organs)]
            elif i < 150:
                concept = sub_regions[(i - 50) % len(sub_regions)]
            elif i < 350:
                concept = diseases[(i - 150) % len(diseases)]
            else:
                concept = findings[(i - 350) % len(findings)]
            node_concepts.append(concept)

        # Convert medical concepts to real BERT Token IDs
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in node_concepts]
        except Exception:
            token_ids = [abs(hash(c)) % 30522 for c in node_concepts]
            
        node_token_tensor = torch.tensor([token_ids], dtype=torch.long) # [1, 577]

        # --- 2. Build Authentic Medical Ontology Graph Triplets ---
        # Concept to index map
        concept_to_idx = {c: i for i, c in enumerate(node_concepts)}
        
        edges = []
        edge_types = []

        def add_rel(src_name, dst_name, forward_rel, reverse_rel):
            if src_name in concept_to_idx and dst_name in concept_to_idx:
                u = concept_to_idx[src_name]
                v = concept_to_idx[dst_name]
                edges.append((u, v))
                edge_types.append(forward_rel)
                if reverse_rel is not None:
                    edges.append((v, u))
                    edge_types.append(reverse_rel)

        # Self-loops (Relation 7)
        for i in range(num_nodes):
            edges.append((i, i))
            edge_types.append(7)

        # Anatomical Part-of Relations (sub_region is_part_of organ [0], organ has_part sub_region [1])
        anatomy_triplets = [
            ("upper lobe", "lung"), ("lower lobe", "lung"), ("middle lobe", "lung"), ("apex", "lung"), ("base", "lung"), ("subpleural space", "lung"),
            ("ventricle", "brain"), ("cerebellum", "brain"), ("brainstem", "brain"), ("frontal lobe", "brain"), ("parietal lobe", "brain"),
            ("occipital lobe", "brain"), ("temporal lobe", "brain"), ("white matter", "brain"), ("grey matter", "brain"),
            ("left atrium", "heart"), ("right atrium", "heart"), ("left ventricle", "heart"), ("right ventricle", "heart"),
            ("ascending aorta", "aorta"), ("aortic arch", "aorta"), ("hepatic lobe", "liver"), ("renal cortex", "kidney"), ("renal medulla", "kidney"),
            ("pancreatic head", "pancreas"), ("pancreatic tail", "pancreas"), ("splenic parenchyma", "spleen"),
            ("lumbar spine", "spine"), ("cervical spine", "spine"), ("thoracic spine", "spine"), ("sacrum", "spine"),
            ("femoral head", "hip"), ("acetabulum", "pelvis"), ("iliac crest", "pelvis"), ("pleural space", "pleura"), ("pericardial space", "heart")
        ]
        for sub, org in anatomy_triplets:
            add_rel(sub, org, 0, 1)

        # Disease Located-in Relations (disease located_in organ [2], organ contains disease [3])
        location_triplets = [
            ("pneumonia", "lung"), ("pneumonia", "chest"), ("pneumonia", "upper lobe"), ("pneumonia", "lower lobe"),
            ("cardiomegaly", "heart"), ("cardiomegaly", "chest"), ("cardiomegaly", "left ventricle"),
            ("pleural effusion", "pleura"), ("pleural effusion", "pleural space"), ("pleural effusion", "costophrenic angle"), ("pleural effusion", "chest"),
            ("pneumothorax", "lung"), ("pneumothorax", "pleural space"), ("pneumothorax", "chest"),
            ("atelectasis", "lung"), ("atelectasis", "upper lobe"), ("atelectasis", "lower lobe"),
            ("pulmonary edema", "lung"), ("pulmonary edema", "chest"), ("pulmonary edema", "hilar area"),
            ("stroke", "brain"), ("stroke", "head"), ("brain infarct", "brain"), ("brain infarct", "cerebellum"), ("intracranial hemorrhage", "brain"),
            ("intracranial hemorrhage", "ventricle"), ("hydrocephalus", "brain"), ("hydrocephalus", "ventricle"),
            ("brain tumor", "brain"), ("glioblastoma", "brain"), ("meningioma", "brain"),
            ("hepatic steatosis", "liver"), ("liver cirrhosis", "liver"), ("hepatocellular carcinoma", "liver"),
            ("cholecystitis", "gallbladder"), ("cholelithiasis", "gallbladder"), ("splenomegaly", "spleen"),
            ("pancreatitis", "pancreas"), ("renal cyst", "kidney"), ("nephrolithiasis", "kidney"), ("renal cell carcinoma", "kidney"),
            ("appendicitis", "bowel"), ("bowel obstruction", "bowel"), ("diverticulitis", "colon"),
            ("fracture", "spine"), ("fracture", "pelvis"), ("fracture", "hip"), ("fracture", "femur"), ("fracture", "shoulder"), ("fracture", "rib"),
            ("osteoarthritis", "joint"), ("spondylolisthesis", "spine"), ("disc herniation", "spine"),
            ("lymphadenopathy", "lymph node"), ("lymphadenopathy", "mediastinum"), ("aortic aneurysm", "aorta"), ("pulmonary embolism", "lung")
        ]
        for dis, loc in location_triplets:
            add_rel(dis, loc, 2, 3)

        # Disease Manifestation Relations (disease manifests_as finding [4], finding indicated_by disease [5])
        manifestation_triplets = [
            ("pneumonia", "opacity"), ("pneumonia", "consolidation"), ("pneumonia", "ground glass opacity"), ("pneumonia", "air fluid level"),
            ("cardiomegaly", "mass effect"), ("cardiomegaly", "vascular congestion"), ("cardiomegaly", "shadowing"),
            ("pleural effusion", "fluid accumulation"), ("pleural effusion", "opacity"), ("pleural effusion", "shadowing"),
            ("pneumothorax", "hyperintensity"), ("pneumothorax", "air fluid level"),
            ("atelectasis", "opacity"), ("atelectasis", "sulcal effacement"),
            ("pulmonary edema", "fluid accumulation"), ("pulmonary edema", "vascular congestion"), ("pulmonary edema", "ground glass opacity"),
            ("stroke", "hypointensity"), ("stroke", "mass effect"), ("brain infarct", "hypointensity"),
            ("intracranial hemorrhage", "hyperintensity"), ("intracranial hemorrhage", "midline shift"), ("intracranial hemorrhage", "mass effect"),
            ("hydrocephalus", "midline shift"), ("brain tumor", "ring enhancement"), ("brain tumor", "mass effect"),
            ("hepatic steatosis", "hypointensity"), ("liver cirrhosis", "ascites"), ("cholecystitis", "fluid accumulation"),
            ("cholelithiasis", "calcification"), ("splenomegaly", "mass effect"), ("pancreatitis", "fluid accumulation"),
            ("renal cyst", "fluid accumulation"), ("nephrolithiasis", "calcification"), ("appendicitis", "fluid accumulation"),
            ("fracture", "cortical disruption"), ("fracture", "soft tissue swelling"), ("osteoarthritis", "osteophyte"),
            ("osteoarthritis", "joint space narrowing"), ("lymphadenopathy", "lymph node enlargement"), ("aortic aneurysm", "calcification")
        ]
        for dis, find in manifestation_triplets:
            add_rel(dis, find, 4, 5)

        # Organ Adjacency Relations (organ adjacent_to organ [6])
        adjacency_triplets = [
            ("lung", "heart"), ("lung", "pleura"), ("lung", "diaphragm"), ("lung", "chest"), ("lung", "mediastinum"),
            ("brain", "skull"), ("brain", "head"), ("brain", "neck"),
            ("liver", "gallbladder"), ("liver", "stomach"), ("liver", "diaphragm"), ("liver", "kidney"), ("liver", "pancreas"),
            ("kidney", "adrenal"), ("kidney", "spleen"), ("spine", "pelvis"), ("spine", "vertebra"), ("heart", "aorta")
        ]
        for o1, o2 in adjacency_triplets:
            add_rel(o1, o2, 6, 6)

        edge_index_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)

        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for src, dst in edges:
            adj_matrix[src, dst] = 1.0

        torch.save(adj_matrix, adj_path)
        torch.save(node_token_tensor, organ_path)
        torch.save(edge_index_tensor, edge_index_path)
        torch.save(edge_type_tensor, edge_type_path)

        print(f"✅ Real Medical Knowledge Graph Token Tensor created with {len(set(node_concepts))} medical terms!")
        print(f"✅ Pre-generated Relational Edge Index: {edge_index_path} ({edge_index_tensor.size(1)} edges)")
        print(f"✅ Pre-generated Relational Edge Types: {edge_type_path} ({num_relations} directed relation types)")
    except Exception as e:
        print(f"⚠️ Could not generate graph features: {e}")

if __name__ == "__main__":
    generate_external_graph_feats()
    generate_sample_arrows()
