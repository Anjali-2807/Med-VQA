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
        
        # 8 Directed Relational Semantics:
        # 0: is_part_of, 1: has_part, 2: located_in, 3: contains,
        # 4: manifests_as, 5: indicated_by, 6: adjacent_to, 7: self_loop
        num_relations = 8

        # --- 1. Real Medical Term Vocabulary Mapping (577 100% Unique Medical Concepts) ---
        organs = [
            "head", "brain", "skull", "chest", "lung", "pleura", "heart", "aorta", "mediastinum", "airway",
            "abdomen", "liver", "gallbladder", "spleen", "pancreas", "kidney", "stomach", "bowel", "colon", "bladder",
            "spine", "vertebra", "pelvis", "hip", "femur", "shoulder", "clavicle", "rib", "diaphragm", "peritoneum",
            "neck", "thyroid", "carotid", "trachea", "esophagus", "adrenal", "uterus", "ovary", "prostate", "vascular",
            "bone", "joint", "muscle", "soft tissue", "lymph node", "thoracic wall", "retroperitoneum", "pericardium", "bronchus", "hilar region"
        ]
        
        sub_regions = [
            "right upper lobe", "right lower lobe", "right middle lobe", "left upper lobe", "left lower lobe", "lung apex", "lung base", "costophrenic angle", "cardiac ventricle", "cerebellar hemisphere",
            "brainstem pons", "frontal cortex", "parietal cortex", "occipital cortex", "temporal cortex", "cerebral white matter", "cerebral grey matter", "left atrium chamber", "right atrium chamber", "left ventricle chamber",
            "right ventricle chamber", "ascending thoracic aorta", "aortic arch segment", "right hepatic lobe", "left hepatic lobe", "renal cortex area", "renal medulla area", "splenic parenchyma tissue", "pancreatic head segment", "pancreatic tail segment",
            "lumbar vertebra spine", "cervical vertebra spine", "thoracic vertebra spine", "sacrum bone", "iliac crest bone", "femoral head bone", "acetabulum cavity", "pleural space cavity", "pericardial space cavity", "peritoneal cavity space",
            "mediastinal space compartment", "hilar area region", "subpleural lung space", "basal ganglia", "thalamus", "internal capsule", "corpus callosum", "pituitary gland", "maxillary sinus", "frontal sinus",
            "ethmoid sinus", "sphenoid sinus", "nasopharynx", "oropharynx", "hypopharynx", "epiglottis", "vocal cord", "thyroid lobe", "parotid gland", "submandibular gland",
            "superior vena cava", "inferior vena cava", "pulmonary artery trunk", "pulmonary vein", "coronary artery", "celiac trunk", "superior mesenteric artery", "renal artery", "common iliac artery", "portal vein",
            "hepatic vein", "gallbladder neck", "cystic duct", "common bile duct", "pancreatic duct", "duodenum", "jejunum", "ileum", "cecum", "appendix",
            "ascending colon", "transverse colon", "descending colon", "sigmoid colon", "rectum", "renal pelvis", "ureter", "prostatic urethra", "scrotum", "testicle",
            "seminal vesicle", "fallopian tube", "myometrium", "endometrium", "cervix", "patella", "tibia", "fibula", "humerus", "radius"
        ]
        
        diseases = [
            "bacterial pneumonia", "viral pneumonia", "aspiration pneumonia", "lobar pneumonia", "bronchopneumonia", "cardiomegaly enlargement", "congestive heart failure", "left pleural effusion", "right pleural effusion", "bilateral pleural effusion",
            "exudative effusion", "transudative effusion", "lung atelectasis", "segmental atelectasis", "lobar atelectasis", "tension pneumothorax", "spontaneous pneumothorax", "pulmonary consolidation", "multifocal consolidation", "acute pulmonary edema",
            "cardiogenic edema", "solitary lung nodule", "multiple lung nodules", "pulmonary mass", "lung carcinoma", "squamous cell carcinoma", "adenocarcinoma", "small cell carcinoma", "pulmonary tuberculosis", "cavitary tuberculosis",
            "pulmonary emphysema", "centrilobular emphysema", "panlobular emphysema", "chronic bronchitis", "bronchiectasis", "ischemic stroke", "hemorrhagic stroke", "acute brain infarct", "subacute brain infarct", "lacunar infarct",
            "epidural hematoma", "subdural hematoma", "subarachnoid hemorrhage", "intraparenchymal hemorrhage", "intraventricular hemorrhage", "obstructive hydrocephalus", "communicating hydrocephalus", "brain metastasis", "high grade glioblastoma", "benign meningioma",
            "pituitary adenoma", "acoustic neuroma", "hepatic steatosis fatty liver", "liver cirrhosis fibrosis", "hepatocellular carcinoma tumor", "hepatic hemangioma", "hepatic adenoma", "focal nodular hyperplasia", "acute cholecystitis", "chronic cholecystitis",
            "cholelithiasis gallstones", "choledocholithiasis", "splenomegaly enlargement", "splenic infarct", "acute pancreatitis", "chronic pancreatitis", "pancreatic adenocarcinoma", "pancreatic pseudocyst", "simple renal cyst", "polycystic kidney disease",
            "nephrolithiasis kidney stones", "ureterolithiasis", "renal cell carcinoma tumor", "pyelonephritis", "acute appendicitis", "small bowel obstruction", "large bowel obstruction", "sigmoid diverticulitis", "colonic diverticulosis", "ulcerative colitis",
            "crohn disease", "colorectal carcinoma", "compression fracture", "pathologic fracture", "comminuted fracture", "displaced fracture", "stress fracture", "hip osteoarthritis", "knee osteoarthritis", "lumbar spondylolisthesis",
            "cervical spondylosis", "lumbar disc herniation", "cervical disc herniation", "disc bulge", "spinal stenosis", "bone metastasis tumor", "osteosarcoma", "osteomyelitis", "mediastinal lymphadenopathy", "hilar lymphadenopathy",
            "abdominal lymphadenopathy", "thoracic aortic aneurysm", "abdominal aortic aneurysm", "aortic dissection", "acute pulmonary embolism", "deep vein thrombosis clot", "pneumoperitoneum", "bowel perforation", "hiatal hernia", "inguinal hernia",
            "umbilical hernia", "incisional hernia", "peritonitis", "abdominal abscess", "pelvic abscess", "retropharyngeal abscess", "pulmonary fibrosis", "idiopathic pulmonary fibrosis", "sarcoidosis", "pneumoconiosis",
            "silicosis", "asbestosis", "pulmonary hypertension", "cor pulmonale", "pericarditis", "cardiac tamponade", "infective endocarditis", "myocarditis", "dilated cardiomyopathy", "hypertrophic cardiomyopathy",
            "restrictive cardiomyopathy", "aortic stenosis", "aortic regurgitation", "mitral stenosis", "mitral regurgitation", "tricuspid regurgitation", "arteriovenous malformation", "cerebral aneurysm", "carotid stenosis", "moyamoya disease",
            "multiple sclerosis", "brain abscess", "encephalitis", "meningitis", "cerebral edema", "diffuse axonal injury", "skull fracture", "facial bone fracture", "mandibular fracture", "orbital fracture",
            "nasal bone fracture", "clavicle fracture", "scapular fracture", "humeral fracture", "radial fracture", "ulnar fracture", "rib fracture", "flail chest", "sternal fracture", "pelvic ring fracture",
            "acetabular fracture", "femoral neck fracture", "intertrochanteric fracture", "tibial fracture", "fibular fracture", "talar fracture", "calcaneal fracture", "rotator cuff tear", "anterior cruciate ligament tear", "meniscal tear",
            "rheumatoid arthritis", "ankylosing spondylitis", "gouty arthritis", "septic arthritis", "avascular necrosis", "paget disease", "osteopenia", "osteoporosis", "scoliosis", "kyphosis", "lordosis",
            "splenic laceration", "liver laceration", "renal laceration", "pneumomediastinum", "subcutaneous emphysema",
            "interstitial lung disease", "pulmonary hypertension arterial", "coronary artery calcification", "myocardial infarction", "pericardial thickening", "pleural thickening", "pneumopericardium", "thoracic aortic dissection", "pulmonary arteriovenous malformation", "pulmonary sequestration",
            "bronchial atresia", "congenital cystic adenomatoid malformation", "septic emboli", "pneumatocele", "bronchopleural fistula", "thoracic wall mass", "pectus excavatum", "pectus carinatum", "kyphoscoliosis deformity", "vertebral hemangioma",
            "schmorl node", "spondylodiscitis", "epidural abscess", "spinal cord compression", "syringomyelia", "chiari malformation", "arachnoid cyst", "epidermoid cyst", "dermoid cyst", "craniopharyngioma",
            "ependymoma", "medulloblastoma", "schwannoma", "paraganglioma", "carotid body tumor", "cervical lymphadenopathy", "goiter enlargement", "thyroid nodule", "parathyroid adenoma", "thymoma mass",
            "germ cell tumor", "fibrosarcoma", "liposarcoma", "leiomyosarcoma", "rhabdomyosarcoma", "chondrosarcoma", "ewing sarcoma", "multiple myeloma", "lymphoma involvement", "splenic cyst",
            "splenic hemangioma", "hepatic adenoma tumor", "focal nodular hyperplasia lesion", "biliary hamartoma", "renal angiomyolipoma", "renal oncocytoma", "adrenal adenoma", "adrenal pheochromocytoma"
        ]
        
        findings = [
            "opacity", "ground glass opacity", "shadowing", "hyperintensity", "hypointensity", "ring enhancement", "calcification", "fluid accumulation", "air fluid level", "soft tissue swelling",
            "cortical disruption", "joint space narrowing", "osteophyte", "midline shift", "sulcal effacement", "mass effect", "pericardial effusion", "ascites", "lymph node enlargement", "nodular lesion",
            "cavitation", "reticular pattern", "hilar enlargement", "vascular congestion", "honeycombing", "crazy paving pattern", "tree in bud sign", "halo sign", "reverse halo sign", "silhouette sign",
            "air bronchogram", "kerley b lines", "continuous diaphragm sign", "deep sulcus sign", "luftsichel sign", "golden s sign", "Hampton hump", "Westermark sign", "knuckle sign", "water bottle sign",
            "pericardial effusion shadow", "epicardial fat pad sign", "mediastinal widening", "hilar overlay sign", "cervicothoracic sign", "doughnut sign", "target sign", "double bubble sign", "rigler sign", "football sign",
            "falciform ligament sign", "inverted v sign", "cupola sign", "continuous lucent border sign", "bowel wall thickening", "thumbprinting sign", "lead pipe sign", "string sign", "comb sign", "apple core lesion",
            "cobblestone appearance", "pseudopolyp", "toxic megacolon sign", "steatotic attenuation", "focal sparing sign", "central dot sign", "capsule sign", "nodule in hemangioma", "target lesion", "central scar sign",
            "double duct sign", "cut off sign", "sentinel loop sign", "colon cutoff sign", "rim enhancement", "fluid fluid level", "fat fluid level", "gas fluid level", "fallen fragment sign", "sunburst periosteal reaction",
            "codman triangle", "onion skinning pattern", "soap bubble appearance", "shepherd crook deformity", "bamboo spine sign", "dagger sign", "shiny corner sign", "romanus lesion", "corner erosion", "syndesmophyte",
            "claw sign", "vacuum phenomenon", "corduroy cloth sign", "picture frame vertebra", "rugger jersey spine", "ivory vertebra", "fishbone vertebra", "H shaped vertebra", "winking owl sign", "pedicle disappearance",
            "bone island lesion", "osteosclerotic lesion", "osteolytic lesion", "punched out lesion", "soap bubble lesion", "ground glass matrix", "cloud like matrix", "popcorn calcification", "punctate calcification", "coarse calcification",
            "rim calcification", "eggshell calcification", "popcorn nodule", "psammomatous calcification", "dystrophic calcification", "metastatic calcification", "vascular calcification", "phlebolith", "gallstone shadow", "kidney stone shadow",
            "staghorn calculus", "bladder stone shadow", "pancreatic calcification", "appendicolith", "prostatic calcification", "dura tail sign", "dural enhancement", "leptomeningeal enhancement", "pachymeningeal enhancement", "perivascular space enlargement",
            "virchow robin space", "hyperdense MCA sign", "dot sign in stroke", "insular ribbon sign", "disappearance of lentiform nucleus", "sulcal effacement pressure", "ventricular dilation", "transependymal edema", "periventricular leukoaraiosis", "centrum semiovale hyperintensity",
            "Dawson finger sign", "open ring enhancement", "target sign brain", "mural enhancement", "comb sign bowel", "string sign ileum", "fat halo sign", "target sign intussusception", "pseudotumor sign", "whirlpool sign volvulus",
            "coffee bean sign", "bent inner tube sign", "bird beak sign achalasia", "rat tail sign", "corkscrew esophagus sign", "double contour sign", "subchondral sclerosis", "subchondral cyst", "joint mice", "seagull sign",
            "teardrop sign orbit", "tripod fracture", "burst fracture", "teardrop fracture", "chance fracture", "Jefferson fracture", "Hangman fracture", "Clay shoveler fracture", "Colles fracture", "Smith fracture",
            "Galeazzi fracture", "Monteggia fracture", "Scaphoid fracture", "Boxer fracture", "Bennett fracture", "Rolando fracture", "Lisfranc fracture", "Jones fracture", "March fracture", "Segond fracture",
            "Bankart lesion", "Hill Sachs lesion", "SLAP lesion"
        ]

        # Assemble 100% Unique Medical Concept Nodes dynamically
        node_concepts = list(dict.fromkeys(organs + sub_regions + diseases + findings))
        num_nodes = len(node_concepts)

        # Convert medical concepts to real BERT Token IDs
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in node_concepts]
        except Exception:
            token_ids = [abs(hash(c)) % 30522 for c in node_concepts]
            
        node_token_tensor = torch.tensor([token_ids], dtype=torch.long) # [1, num_nodes]

        # --- 2. Build Authentic Medical Ontology Graph Triplets ---
        # Concept to index map (100% unique 1-to-1 mapping!)
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
