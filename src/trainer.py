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
        input_text, teacher_input_text, input_audio, teacher_input_audio, label = batch
        
        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)

        # Forward pass
        output = self.network(input_text, input_audio)
        loss = self.criterion(output[0], label)
        
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
        input_text, teacher_input_text, input_audio, teacher_input_audio, label = batch
        
        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss = self.criterion(output[0], label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

class DistilTrainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        teacher: _4M_SER,
        network: _4M_SER,
        label_criterion: torch.nn.CrossEntropyLoss = None,
        distil_criterion: torch.nn.KLDivLoss = None,
        feature_criterion: torch.nn.MSELoss = None,
        alpha: float = 0.5,
        T: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.teacher = teacher
        self.network = network
        self.label_criterion = label_criterion
        self.distil_criterion = distil_criterion
        self.feature_criterion = feature_criterion
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
        input_text, teacher_input_text, input_audio, teacher_input_audio, label = batch

        # Move inputs to cpu or gpu
        input_text = input_text.to(self.device)
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        teacher_input_text = teacher_input_text.to(self.device)
        teacher_input_audio = teacher_input_audio.to(self.device)

        # Forward pass
        with torch.no_grad():
            teacher_output = self.teacher(teacher_input_text, teacher_input_audio)
            
        student_output = self.network(input_text, input_audio)
        
        # Calculate the feature loss using MSE
        feature_loss = self.text_criterion(teacher_output[2], student_output[2])
        
        # Calculate the soft targets loss using KL divergence.
        distil_loss = self.distil_criterion(student_output[0], teacher_output[0])
        
        # Calculate the label loss using cross entropy
        label_loss = self.label_criterion(student_output[0], label)
        
        # L(total) = (alpha)L(soft_targets) + (1-alpha)L(label)
        total_loss = self.alpha * distil_loss + (1 - self.alpha) * label_loss + feature_loss
        
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
        input_text, teacher_input_text, input_audio, teacher_input_audio, label = batch
        
        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss = self.criterion(output[0], label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }