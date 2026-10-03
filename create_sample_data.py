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
        
        if not os.path.exists(adj_path):
            torch.save(torch.eye(577, dtype=torch.float32), adj_path)
            print(f"✅ Pre-generated Knowledge Graph adjacency tensor: {adj_path}")
            
        if not os.path.exists(organ_path):
            torch.save(torch.randint(0, 30522, (1, 577), dtype=torch.long), organ_path)
            print(f"✅ Pre-generated Knowledge Graph token tensor: {organ_path}")
    except Exception as e:
        print(f"⚠️ Could not generate graph features: {e}")

if __name__ == "__main__":
    generate_external_graph_feats()
    generate_sample_arrows()
