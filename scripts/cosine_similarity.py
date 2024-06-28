import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
import csv
import glob
import argparse
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
from data.dataloader import build_train_test_dataset
from tqdm.auto import tqdm
from models import networks
from configs.base import Config
from collections import Counter
from typing import Tuple
from transformers import logging as logg
import torch.nn as nn

logg.set_verbosity_error()

def cosine_similarity(a, b):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(a, b)

def consine_similarity(teacher_cfg, cfg, teacher_checkpoint_path, checkpoint_path, all_state_dict=True, cm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = getattr(networks, cfg.model_type)(cfg)
    teacher = getattr(networks, teacher_cfg.model_type)(teacher_cfg)
    
    student.to(device)

    # Build dataset
    _, test_ds = build_train_test_dataset(cfg)
    
    teacher_weight = torch.load(teacher_checkpoint_path, map_location=torch.device(device))
    teacher.load_state_dict(teacher_weight)
    teacher.eval()
    teacher.to(device)
    
    
    student_weight = torch.load(checkpoint_path, map_location=torch.device(device))
    if all_state_dict:
        student_weight = student_weight["state_dict_network"]
    student.load_state_dict(student_weight)
    student.eval()
    student.to(device)

    cos_sim = []

    for every_test_list in tqdm(test_ds):
        input_ids, _, audio, _, label = every_test_list
        input_ids = input_ids.to(device)
        audio = audio.to(device)
        label = label.to(device)
        with torch.no_grad():
            student_fusion_features = student(input_ids, audio)[1]
            teacher_fusion_features = teacher(input_ids, audio)[1]
            
            cos_sim.append(cosine_similarity(student_fusion_features, teacher_fusion_features).item())

    with open(os.path.join(checkpoint_path[: checkpoint_path.find("weights")], f"teacher_{cfg.text_encoder_type}_cos_sim_{cfg.data_valid}"), "wb") as f:
        pickle.dump(cos_sim, f)
    
    return cos_sim

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-teacher_ckpt",
        "--teacher_checkpoint_path",
        type=str,
        default="checkpoints_latest/_4M_SER_bert_vggish3/20240613-015337/weights/best_acc/checkpoint_206034.pth",
        help="path to teacher checkpoint folder",
    )
    
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        default="checkpoints_latest/student_kd/_4M_SER_microbert_vggish/20240624-091622/weights/best_acc/checkpoint_21_94059.pt",
        help="path to checkpoint folder",
    )

    parser.add_argument(
        "-dict",
        "--state_dict",
        action="store_false",
        help="whether to use all state dict or not",
    )

    parser.add_argument(
        "-t",
        "--test_set",
        type=str,
        default="test.pkl",
        help="name of testing set. Ex: test.pkl",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    teacher_ckpt_path = args.teacher_checkpoint_path
    ckpt_path = args.checkpoint_path
    if not os.path.exists(ckpt_path):
        print("Checkpoint path does not exist")

    teacher_cfg_path = os.path.join(teacher_ckpt_path[: teacher_ckpt_path.find("weights")], "cfg.log")
    teacher_cfg = Config()
    teacher_cfg.load(teacher_cfg_path)

    cfg_path = os.path.join(ckpt_path[: ckpt_path.find("weights")], "cfg.log")
    cfg = Config()
    cfg.load(cfg_path)
    # Change to test set
    test_set = args.test_set if args.test_set is not None else "test.pkl"
    cfg.data_valid = test_set

    consine_similarity(teacher_cfg, cfg, teacher_ckpt_path, ckpt_path)
