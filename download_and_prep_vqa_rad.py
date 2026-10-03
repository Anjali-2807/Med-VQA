import os
import random
import io
from PIL import Image

def download_and_preprocess_vqa_rad(save_dir="data/finetune_arrows"):
    os.makedirs(save_dir, exist_ok=True)
    
    needed = ["vqa_vqa_rad_train.arrow", "vqa_vqa_rad_val.arrow", "vqa_vqa_rad_test.arrow"]
    if all(os.path.exists(os.path.join(save_dir, f)) for f in needed):
        # Check if files are non-empty real dataset files (e.g. > 1MB)
        first_file = os.path.join(save_dir, "vqa_vqa_rad_train.arrow")
        if os.path.getsize(first_file) > 1000000:
            print(f"✅ Full VQA-RAD dataset arrow files already present in '{save_dir}'")
            return

    print("📥 Downloading full VQA-RAD dataset from Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install -q datasets")
        from datasets import load_dataset

    hf_dataset = load_dataset("flaviagiammarino/vqa-rad")
    print("✅ Hugging Face VQA-RAD dataset downloaded!")

    img_dir = "data/vqa_rad_images"
    os.makedirs(img_dir, exist_ok=True)

    random.seed(42)
    
    all_samples = []
    sample_id = 0
    for split_name in hf_dataset.keys():
        ds_split = hf_dataset[split_name]
        for idx, item in enumerate(ds_split):
            img = item["image"]
            img_name = f"vqa_rad_{split_name}_{idx}.jpg"
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(img_path, format="JPEG")
            
            q_text = str(item.get("question", "")).strip()
            ans_text = str(item.get("answer", "")).strip()
            ans_type = str(item.get("answer_type", "CLOSED")).strip().upper()
            if ans_type not in ["CLOSED", "OPEN"]:
                ans_type = "CLOSED" if ans_text.lower() in ["yes", "no"] else "OPEN"
            
            all_samples.append({
                "img_path": img_path,
                "qid": sample_id,
                "question": q_text,
                "answer": ans_text,
                "answer_type": ans_type
            })
            sample_id += 1

    print(f"📊 Total VQA-RAD QA samples loaded: {len(all_samples)}")
    random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    data = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train+n_val],
        "test": all_samples[n_train+n_val:]
    }

    print(f"  --> Train samples: {len(data['train'])}")
    print(f"  --> Val samples:   {len(data['val'])}")
    print(f"  --> Test samples:  {len(data['test'])}")

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepro"))
    from prepro.make_arrow import make_arrow_vqa
    from create_sample_data import generate_external_graph_feats
    generate_external_graph_feats()
    
    make_arrow_vqa(data, "vqa_vqa_rad", save_dir)
    print(f"🎉 Full VQA-RAD dataset preprocessed & saved to {save_dir}!")

if __name__ == "__main__":
    download_and_preprocess_vqa_rad()
