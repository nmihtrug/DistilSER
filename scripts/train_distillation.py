import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import nn, optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback
from tqdm.auto import tqdm
from transformers import logging as logg
logg.set_verbosity_error()

SEED = 78
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg: Config, tea_cfg: Config):
    # Teacher model
    logging.info("Initializing teacher model...")
    try:
        teacher = getattr(networks, tea_cfg.model_type)(tea_cfg)
        teacher.to(device)
    except AttributeError:
        raise NotImplementedError("Teacher model {} is not implemented".format(tea_cfg.model_type))

    # Load teacher model from checkpoint
    logging.info("Loading teacher model from checkpoint...")
    try:
        teacher_checkpoint = torch.load(cfg.teacher_checkpoint, map_location=torch.device(device))
        teacher_checkpoint = teacher_checkpoint["state_dict_network"]
        teacher_checkpoint.pop("text_encoder.embeddings.position_ids")
        teacher.load_state_dict(teacher_checkpoint)
    except Exception:
        raise ValueError("Failed to load teacher model from checkpoint {}".format(cfg.teacher_checkpoint))

    # Student model
    logging.info("Initializing student model...")
    try:
        student = getattr(networks, cfg.model_type)(cfg)
        student.to(device)
    except AttributeError:
        raise NotImplementedError("Student model {} is not implemented".format(cfg.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    try:
        label_criterion = getattr(losses, cfg.label_loss_type)(cfg)
        label_criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Label loss {} is not implemented".format(cfg.label_loss_type))

    try:
        feature_criterion = getattr(losses, cfg.fusion_loss_type)(cfg)
        feature_criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Fusion loss {} is not implemented".format(cfg.fusion_loss_type))
    
    try:
        distil_criterion = getattr(losses, cfg.distil_loss_type)(cfg)
        distil_criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Fusion loss {} is not implemented".format(cfg.distil_loss_type))

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            teacher=teacher,
            network=student,
            label_criterion=label_criterion,
            distil_criterion=distil_criterion,
            feature_criterion=feature_criterion,
            alpha=cfg.alpha,
            T=cfg.T,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))


    print(trainer.teacher.state_dict()['text_encoder.encoder.layer.10.attention.self.key.weight'], teacher.state_dict()['text_encoder.encoder.layer.10.attention.self.key.weight'], teacher.state_dict()['text_encoder.encoder.layer.10.attention.self.key.weight'] == trainer.teacher.state_dict()['text_encoder.encoder.layer.10.attention.self.key.weight'])
    return
    # Teacher take encode data as input
    trainer.teacher.transfer_learning = True
    
    if cfg.transfer_learning:
        logging.info("Transfer learning phase")
        trainer.network.transfer_learning = True
        train_ds_encode, test_ds_encode = build_train_test_dataset(cfg, trainer.network, trainer.teacher)
        optimizer_transfer = optims.get_optim(cfg, student)
        trainer.compile(optimizer=optimizer_transfer)
        ckpt_callback_transfer = CheckpointsCallback(
            checkpoint_dir=weight_dir,
            save_freq=cfg.num_transfer_epochs * len(train_ds_encode) * 2,
            max_to_keep=cfg.max_to_keep,
            save_best_val=True,
            save_all_states=False,
        )
        trainer.fit(
            train_ds_encode,
            cfg.num_transfer_epochs,
            test_ds_encode,
            callbacks=[ckpt_callback_transfer],
        )

        trainer.network.load_state_dict(torch.load(ckpt_callback_transfer.best_path))
        trainer.network.transfer_learning = False
        del (
            train_ds_encode,
            test_ds_encode,
            optimizer_transfer,
            ckpt_callback_transfer,
        )

    train_ds, test_ds = build_train_test_dataset(cfg, None, trainer.teacher)
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, student)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )

    logging.info("Fine-tuning phase")
    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-stu_cfg", "--student_config", type=str, default="../src/configs/microbert_vggish.py")
    parser.add_argument("-tea_cfg", "--teacher_config", type=str, default="../src/configs/bert_vggish.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    stu_cfg: Config = get_options(args.student_config)
    tea_cfg: Config = get_options(args.teacher_config)

    if stu_cfg.resume and stu_cfg.cfg_path is not None:
        resume = stu_cfg.resume
        resume_path = stu_cfg.resume_path
        stu_cfg.load(stu_cfg.cfg_path)
        stu_cfg.resume = resume
        stu_cfg.resume_path = resume_path

    main(stu_cfg, tea_cfg)
