import os
import re
import csv
import sys
import json
import fire
import torch
import random
import pprint
import numpy as np
import importlib.util
import logging as log
from datetime import datetime
from tqdm import tqdm
from types import SimpleNamespace # Quan trọng để giả lập object cho LF

from tdpm import tdpm

import wrench
from wrench.dataset import load_dataset
from wrench._logging import LoggingHandler
from wrench.endmodel import EndClassifierModel, LogRegModel
from wrench.labelmodel import Snorkel, MajorityVoting, MajorityWeightedVoting

# Import modules của chúng ta
from auto_lf.extractors.vlm_wrapper import VLMExtractor
from auto_lf.extractors.psg_wrapper import PSGExtractor
from auto_lf.router import Router

class Labeler:

    def __init__(self, args):
        
        self.args = args
        self.dataset = args["dataset"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LM_NAMES = ["MV"]
        self.EM_NAMES = ["MLP"]
        self.total_cost = 0
        self.final_result = {"Dataset" : args["dataset"],
                             "Mode" : args["mode"],
                             "Model" : args["codellm"]}
        if "prior_type" in args:
            self.final_result["Heuristic Mode"] = args["prior_type"]

        # Cache để lưu đặc trưng ảnh, tránh chạy lại VLM nhiều lần
        self.feature_cache = {} 
        # Các model extractors sẽ được khởi tạo lazy (khi cần mới load)
        self.router = None
        self.vlm = None

    # --- HÀM MỚI: Load LF sinh tự động ---
    def load_generated_lfs(self, path="generated_lfs.py"):
        """Load các hàm LF từ file python được sinh tự động"""
        # Nếu path chỉ là tên file, ghép với thư mục hiện tại hoặc thư mục LF
        if not os.path.exists(path):
            # Thử tìm trong thư mục LF saved path nếu không thấy ở root
            alt_path = os.path.join(self.args.get("LF_saving_exact_dir", ""), path)
            if os.path.exists(alt_path):
                path = alt_path
            else:
                # log.warning(f"Generated LFs file not found at {path}")
                return []
            
        spec = importlib.util.spec_from_file_location("generated_lfs", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_lfs"] = module
        spec.loader.exec_module(module)
        
        lfs = []
        for name in dir(module):
            obj = getattr(module, name)
            # Wrench LF thường được decorate, kiểm tra callable và tên hàm
            if callable(obj) and name.startswith("auto_lf"):
                lfs.append(obj)
        
        # log.info(f"Loaded {len(lfs)} Auto-LFs from {path}")
        return lfs
    
    # --- HÀM MỚI: Trích xuất đặc trưng ảnh (Có Cache) ---
    def precompute_image_features(self, dataset):
        """
        Chạy Feature Extraction một lần cho toàn bộ dataset.
        Input: Wrench Dataset
        Output: Dict {image_path: set(features)}
        """
        # 1. Khởi tạo Model nếu chưa có (Lazy Load)
        if self.router is None:
            self.router = Router()
            # Lưu ý: Cấu hình device tại đây
            self.vlm = VLMExtractor(device=self.device)
            # self.psg = PSGExtractor(...) 

        current_cache = {}
        # log.info(f"Pre-computing features for {len(dataset)} images...")
        
        # Dùng tqdm để hiện thanh tiến trình
        for i in tqdm(range(len(dataset)), desc="Extracting Features"):
            # Giả định: dataset của wrench lưu đường dẫn ảnh trong trường "text"
            img_path = dataset.examples[i]["text"] 
            
            # Kiểm tra cache toàn cục
            if img_path in self.feature_cache:
                current_cache[img_path] = self.feature_cache[img_path]
                continue

            try:
                # Kiểm tra file tồn tại
                if not os.path.exists(img_path):
                     # Thử fix path tương đối
                     if os.path.exists(os.path.join(".", img_path)):
                         img_path = os.path.join(".", img_path)
                
                if os.path.exists(img_path):
                    # Logic Router
                    mode = self.router.route(img_path)
                    
                    # Logic Extract
                    if mode == "SIMPLE":
                        feats = self.vlm.extract(img_path)
                    else:
                        # Fallback về VLM nếu chưa cài PSG
                        feats = self.vlm.extract(img_path) 
                    
                    # Lưu dưới dạng set để tìm kiếm O(1)
                    feature_set = set(feats)
                    self.feature_cache[img_path] = feature_set
                    current_cache[img_path] = feature_set
                else:
                    self.feature_cache[img_path] = set()
                    current_cache[img_path] = set()

            except Exception as e:
                # Fallback nếu lỗi đọc ảnh
                # log.error(f"Error extracting {img_path}: {e}")
                self.feature_cache[img_path] = set()
                current_cache[img_path] = set()
                
        return current_cache

    def get_model(self, name):
        if name == "WMV":
            return MajorityWeightedVoting()
        elif name == "MV":
            return MajorityVoting()
        elif name == "Snorkel":
            return Snorkel()
        elif name == "MLP":
            return  EndClassifierModel(backbone='MLP')
        elif name == "LR":
            return LogRegModel()

    def get_labeled_subset(self, dataset, labeled_percentage: float = 0.2, min_num_labeled: int = 100):
        num_dataset = len(dataset)
        num_subset = max(min_num_labeled, int(labeled_percentage * num_dataset))
        
        indices = np.random.choice(num_dataset, size=num_subset, replace=False)
        labeled_subset = dataset.create_subset(indices)
        
        return labeled_subset

    def get_labeled_subset_per_class(self, dataset, per_class: int = 20):
        labels = np.array(dataset.labels)
        unique_labels = np.unique(labels)

        selected_indices = []
        for lbl in unique_labels:
            indices_lbl = np.where(labels == lbl)[0]
            n_select = min(per_class, len(indices_lbl))
            chosen = np.random.choice(indices_lbl, size=n_select, replace=False)
            selected_indices.extend(chosen)

        selected_indices = np.array(selected_indices)
        labeled_subset = dataset.create_subset(selected_indices)
        return labeled_subset

    def get_data(self, label_num=None):
        if label_num is not None:
            train_data, valid_data, test_data = load_dataset(
                self.args['dataset_LF_saved_path'].replace(self.dataset, ""),
                self.dataset,
                extract_feature=True,
                extract_fn='bert',
                model_name='bert-base-cased',
                cache_name='bert',
                label_num=label_num,
            )
        else:
            train_data, valid_data, test_data = load_dataset(
                self.args['dataset_LF_saved_path'].replace(self.dataset, ""),
                self.dataset,
                extract_feature=True,
                extract_fn='bert',
                model_name='bert-base-cased',
                cache_name='bert',
            )
        
        np.random.seed(0)
        valid_data = self.get_labeled_subset(valid_data)
        return train_data, valid_data, test_data

    def sort_filenames(self, filename):
        return int(re.search(r'\d+', filename).group())
                
    def get_LF_file_paths(self):
        file_path_collection = []
        if os.path.exists(self.args["LF_saving_exact_dir"]):
            for f in os.listdir(self.args["LF_saving_exact_dir"]):
                file_path = os.path.join(self.args["LF_saving_exact_dir"], f)
                if os.path.isfile(file_path) and f.endswith(".py"):
                    file_path_collection.append(f)
                    
            file_path_collection = sorted(file_path_collection, key=self.sort_filenames)
            for i, f in enumerate(file_path_collection):
                exact_file_path = os.path.join(self.args["LF_saving_exact_dir"], f)
                file_path_collection[i] = exact_file_path
        
        return file_path_collection
        
    def get_weak_labels(self, data, type_data, generated_lfs=None):
        """
        Tính toán nhãn yếu từ cả LF cũ (file text) và LF mới (Auto-LF)
        """
        module_spec = importlib.util.spec_from_loader("temp_module", loader=None)
        module = importlib.util.module_from_spec(module_spec)

        weak_label_matrix = []

        # 1. Xử lý LFs cũ (Load từ file text)
        if hasattr(self, 'file_path_collection') and self.file_path_collection:
            for file_path in self.file_path_collection:
                # print(f"Read {file_path} for {type_data} data")
                with open(file_path, "r", encoding="utf-8") as f:
                    code_string = f.read()
                
                try:
                    exec(code_string, module.__dict__)
                    sys.modules["temp_module"] = module
                    from temp_module import label_function
                    
                    weak_labels = []
                    for i in range(len(data.examples)):
                        example = data.examples[i]["text"]
                        weak_label = label_function(example)
                        weak_labels.append(weak_label)
                    weak_label_matrix.append(weak_labels)
                except:
                    self.logger.error(f"Error in {file_path}")

        # 2. Xử lý Auto-LFs (Load từ generated_lfs.py)
        if generated_lfs is not None and len(generated_lfs) > 0:
            # print(f"Applying {len(generated_lfs)} Auto-LFs for {type_data} data...")
            # Trích xuất đặc trưng (chỉ chạy 1 lần nhờ cache)
            feature_map = self.precompute_image_features(data)

            for lf_func in generated_lfs:
                weak_labels = []
                for i in range(len(data.examples)):
                    img_path = data.examples[i]["text"]
                    
                    # Tạo đối tượng giả lập chứa features để pass vào LF
                    feats = feature_map.get(img_path, set())
                    x_proxy = SimpleNamespace(features=feats)
                    
                    try:
                        # Gọi hàm LF
                        wl = lf_func(x_proxy)
                        weak_labels.append(wl)
                    except Exception as e:
                        weak_labels.append(-1) # ABSTAIN nếu lỗi
                
                weak_label_matrix.append(weak_labels)

        # Chuyển vị ma trận: (N_examples, N_LFs)
        if len(weak_label_matrix) == 0:
            return np.array([])
        
        weak_label_matrix = np.array(weak_label_matrix).T
        return weak_label_matrix

    def get_LF_summary(self):
        self.logger.info(f"Training Data LF summary:\n{self.train_data.lf_summary()}")
        self.logger.info(f"Validation Data LF summary:\n{self.valid_data.lf_summary()}")
        self.logger.info(f"Testing Data LF summary:\n{self.test_data.lf_summary()}")
        
    def label_time(self):
        ## filter out uncovered training data ##
        train_data_covered = self.train_data.get_covered_subset()
        if len(self.train_data) > 0:
            lm_coverage = len(train_data_covered) / len(self.train_data)
        else:
            lm_coverage = 0
            
        self.final_result["lm_coverage"] = lm_coverage
        self.logger.info(f'label model train coverage: {round(lm_coverage, 5)}')

        ## run label model for 5 times ##
        TIMES = 5
        for label_model_name in self.LM_NAMES:
            
            self.logger.info("=====================================")
            lm_acc_array = np.zeros(TIMES)
            lm_f1_array = np.zeros(TIMES)
            lm_collection = []
            
            for T1 in range(TIMES):    
                label_model = self.get_model(label_model_name)
                label_model.fit(dataset_train=self.train_data, 
                                dataset_valid=self.valid_data,
                                metric="f1_macro",
                                )
                
                lm_acc = label_model.test(self.test_data, 'acc')
                lm_f1 = label_model.test(self.test_data, 'f1_macro')

                lm_acc_array[T1] = lm_acc
                lm_f1_array[T1] = lm_f1
                lm_collection.append(label_model)
                
                self.logger.info(f'{T1} - {label_model_name} testing accuracy: {round(lm_acc, 5)}')
                self.logger.info(f'{T1} - {label_model_name} testing f1: {round(lm_f1, 5)}')

            ## Overall Evaluation ##
            self.logger.info("=====================================")
            lm_acc_mean, lm_acc_std = np.mean(lm_acc_array, axis=0), np.std(lm_acc_array, axis=0)
            lm_f1_mean, lm_f1_std = np.mean(lm_f1_array, axis=0), np.std(lm_f1_array, axis=0)
            
            self.logger.info(f'Overall - {label_model_name} testing accuracy mean: {round(lm_acc_mean, 5)}')
            self.final_result[f"{label_model_name}_acc_mean"] = round(lm_acc_mean, 5)
            self.final_result[f"{label_model_name}_acc_std"] = round(lm_acc_std, 5)
            self.final_result[f"{label_model_name}_f1_mean"] = round(lm_f1_mean, 5)
            self.final_result[f"{label_model_name}_f1_std"] = round(lm_f1_std, 5)
            
            ## Use best label model to predict soft label ##
            best_label_model_index = np.argmax(lm_f1_array)
            best_label_model = lm_collection[best_label_model_index]
            
            if len(train_data_covered) > 0:
                train_soft_label_covered = best_label_model.predict(train_data_covered)
                lm_acc_train = best_label_model.test(train_data_covered, 'acc')
                lm_f1_train = best_label_model.test(train_data_covered, 'f1_macro')
            else:
                train_soft_label_covered = []
                lm_acc_train = 0
                lm_f1_train = 0
            
            self.final_result.update({
                f"covered_train_acc": lm_acc_train,
                f"covered_train_f1": lm_f1_train,
            })
            
            ## run end model with soft labels for 5 times ##
            if len(train_data_covered) > 0:
                for end_model_name in self.EM_NAMES:
                    self.logger.info("=====================================")
                    em_acc_array = np.zeros(TIMES)
                    em_f1_array = np.zeros(TIMES)
                    
                    m = "f1_macro"
                        
                    for T2 in range(TIMES):  
                        end_model = self.get_model(end_model_name)
                        end_model.fit(dataset_train=train_data_covered, 
                                      y_train=train_soft_label_covered,
                                      dataset_valid=self.valid_data, 
                                      evaluation_step=50, 
                                      metric=m, 
                                      verbose=True,
                                      device=self.device)
                        
                        em_acc = end_model.test(self.test_data, 'acc')
                        em_f1 = end_model.test(self.test_data, 'f1_macro')
                            
                        em_acc_array[T2] = em_acc
                        em_f1_array[T2] = em_f1
                        
                        self.logger.info(f'{T2} - {label_model_name} + {end_model_name} testing accuracy: {round(em_acc, 5)}')

                    ## Overall Evaluation ##
                    em_acc_mean = np.mean(em_acc_array, axis=0)
                    em_acc_std = np.std(em_acc_array, axis=0)
                    em_f1_mean = np.mean(em_f1_array, axis=0)
                    em_f1_std = np.std(em_f1_array, axis=0)
                    
                    self.final_result[f"{label_model_name}_{end_model_name}_acc_mean"] = round(em_acc_mean, 5)
                    self.final_result[f"{label_model_name}_{end_model_name}_acc_std"] = round(em_acc_std, 5)
                    self.final_result[f"{label_model_name}_{end_model_name}_f1_mean"] = round(em_f1_mean, 5)
                    self.final_result[f"{label_model_name}_{end_model_name}_f1_std"] = round(em_f1_std, 5)

    def get_total_cost(self):
        self.logger.info("=====================================")
        if hasattr(self, 'file_path_collection'):
            for file_path in self.file_path_collection:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_string = f.read()
                pattern = re.compile(r'\$\[(.*?)\]')
                matches = pattern.findall(code_string)
                if matches:
                    self.total_cost += float(matches[0])
        self.final_result["total_cost"] = round(self.total_cost, 5)
        self.logger.info(f'Total cost of LFs: ${round(self.total_cost, 5)}')

    def run(self, label_num=None):
        self.logger = log.getLogger(__name__)
        ## get training, validation, testing data with bert features ##
        self.train_data, self.valid_data, self.test_data = self.get_data(label_num)

        ## get LF file paths (Old LFs) ##
        self.file_path_collection = self.get_LF_file_paths()
        
        ## get Auto-LFs (New LFs) ##
        self.auto_lfs = self.load_generated_lfs(path="generated_lfs.py")
        
        total_lfs = len(self.file_path_collection) + len(self.auto_lfs)
        self.final_result["num_of_LF"] = total_lfs
        
        ## produce weak labels (Combine both) ##
        self.logger.info("Computing weak labels for Training Data...")
        self.train_data.weak_labels = self.get_weak_labels(self.train_data, type_data="train", generated_lfs=self.auto_lfs)
        
        self.logger.info("Computing weak labels for Validation Data...")
        self.valid_data.weak_labels = self.get_weak_labels(self.valid_data, type_data="valid", generated_lfs=self.auto_lfs)
        
        self.logger.info("Computing weak labels for Testing Data...")
        self.test_data.weak_labels = self.get_weak_labels(self.test_data, type_data="test", generated_lfs=self.auto_lfs)
        
        ## log result saved place ##
        if not os.path.exists(self.args["exp_result_saved_path"]):
            os.mkdir(self.args["exp_result_saved_path"])
        
        if self.args["mode"] == "ScriptoriumWS":
            log_file_name = os.path.join(self.args["exp_result_saved_path"], self.args["mode"] + "_" + self.args["codellm"] + ".txt")
        else:
            log_file_name = os.path.join(self.args["exp_result_saved_path"], self.args["mode"] + "_" + self.args["prior_type"] + "_" + self.args["codellm"] + ".txt")
    
        ## get logger ##
        for handler in log.root.handlers[:]:
            log.root.removeHandler(handler)
            
        log.basicConfig(level=log.INFO, format='%(asctime)s : %(levelname)s : %(message)s', \
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[log.FileHandler(log_file_name, mode='w'), LoggingHandler()])

        self.logger.info(f'Running {self.dataset} Dataset')
        self.logger.info(f'Running with {total_lfs} labeling funcions')

        ## get LF summary ##
        self.get_LF_summary()
        
        ## run label model and end model ##
        self.label_time()

        ## get total cost of LFs ##
        self.get_total_cost()

        if self.args["mode"] == "ScriptoriumWS":
            self.final_result["Heuristic Mode"] = None
        else:
            self.final_result["Heuristic Mode"] = self.args["prior_type"]
        
        with open('temp_for_copy_comb.csv', 'a', newline='') as f:
            w = csv.DictWriter(f, self.final_result.keys())
            if f.tell() == 0:
                w.writeheader()
            w.writerow(self.final_result)
        return self.final_result