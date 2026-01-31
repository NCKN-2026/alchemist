# run_lf_gen.py
import os
import glob
import json
from auto_lf.router import Router
from auto_lf.extractors.vlm_wrapper import VLMExtractor
# from auto_labeling.extractors.psg_wrapper import PSGExtractor (Uncomment khi cấu hình xong PSG)
from auto_lf.miners.rule_miner import RuleMiner

def main():
    
    with open("data/devset/dev.json", "r") as f:
        dev_data = json.load(f)
    
    image_paths = [item["path"] for item in dev_data]
    dev_labels = [item["label"] for item in dev_data]
    
    router = Router()
    vlm = VLMExtractor() 
    
    all_features = []
    
    for img_path in image_paths:
        mode = router.route(img_path)
        
        if mode == "SIMPLE":
            feats = vlm.extract(img_path)
        else:
            # feats = psg.extract(img_path)
            feats = vlm.extract(img_path) # Tạm dùng VLM fallback
            
        all_features.append(feats)
        
    print("Mining Rules...")
    miner = RuleMiner()
    miner.fit_and_generate(all_features, dev_labels, output_file="generated_lfs.py")
    
    print("Done! Check 'generated_lfs.py'")

if __name__ == "__main__":
    main()