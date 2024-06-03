import logging
import os
from typing import Dict

import torch
from torch import Tensor
from configs.base import Config
from models.networks import _4M_SER
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: _4M_SER,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)

        # Forward pass
        output = self.network(input_text, input_audio)
        loss = self.criterion(output, label)
        # print(output)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }


class MarginTrainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: _4M_SER,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        self.opt_criterion = torch.optim.Adam(
            params=[{"params": self.criterion.parameters()}],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta_1, cfg.adam_beta_2),
            eps=cfg.adam_eps,
            weight_decay=cfg.adam_weight_decay,
        )

        self.scheduler_criterion = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.opt_criterion,
            lr_lambda=lambda epoch: (
                ((epoch + 1) / (4 + 1)) ** 2
                if epoch < -1
                else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
            ),
        )

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()
        self.opt_criterion.zero_grad()

        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)

        # Forward pass
        output = self.network(input_text, input_audio)
        loss, logits = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.opt_criterion.step()

        # Calculate accuracy
        _, preds = torch.max(logits, 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss, logits = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(logits, 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def lr_scheduler(self, step: int, epoch: int):
        if self.scheduler is not None:
            self.scheduler.step()
        self.scheduler_criterion.step()

    def save_all_states(self, path: str, global_epoch: int, global_step: int):
        checkpoint = {
            "epoch": global_epoch,
            "global_step": global_step,
            "state_dict_network": self.network.state_dict(),
            "state_optimizer": self.optimizer.state_dict(),
            "state_criterion": self.criterion.state_dict(),
            "state_lr_scheduler_criterion": self.scheduler_criterion.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["state_lr_scheduler"] = self.scheduler.state_dict()

        ckpt_path = os.path.join(
            path, "checkpoint_{}_{}.pt".format(global_epoch, global_step)
        )
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_all_states(self, path: str, device: str = "cpu"):
        dict_checkpoint = torch.load(os.path.join(path), map_location=device)

        self.start_epoch = dict_checkpoint["epoch"]
        self.global_step = dict_checkpoint["global_step"]
        self.network.load_state_dict(dict_checkpoint["state_dict_network"])
        self.optimizer.load_state_dict(dict_checkpoint["state_optimizer"])
        self.criterion.load_state_dict(dict_checkpoint["state_criterion"])
        self.scheduler_criterion.load_state_dict(
            dict_checkpoint["state_lr_scheduler_criterion"]
        )
        if self.scheduler is not None:
            self.scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

        logging.info("Successfully loaded checkpoint from {}".format(path))
        logging.info("Resume training from epoch {}".format(self.start_epoch))
        

class DistilTrainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        teacher: _4M_SER,
        network: _4M_SER,
        criterion: torch.nn.CrossEntropyLoss = None,
        fusion_criterion: torch.nn.MSELoss = None,
        alpha: float = 0.1,
        T: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.teacher = teacher
        self.network = network
        self.criterion = criterion
        self.fusion_criterion = fusion_criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher.to(self.device)
        self.network.to(self.device)
        self.alpha = alpha
        self.T = T

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.teacher.eval()  # Teacher set to evaluation mode
        self.network.train() # Student to train mode
        
        self.optimizer.zero_grad()

        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)

        # Forward pass
        with torch.no_grad():
            teacher_output = self.teacher(input_text, input_audio)
            
        student_output = self.network(input_text, input_audio)
        
        # Calculate the fusion loss using MSE
        fusion_loss = self.fusion_criterion(teacher_output[1], student_output[1])
        
        # Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_output[0] / self.T, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_output[0] / self.T, dim=-1)
        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T**2)
        
        label_loss = self.criterion(student_output, label)
        
        # L(total) = (alpha)L(soft_targets) + (1-alpha)L(label) + L(fusion)
        total_loss = self.alpha * soft_targets_loss + (1 - self.alpha) * label_loss + fusion_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(student_output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": total_loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }