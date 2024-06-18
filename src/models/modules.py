import math
import torch
import torch.nn as nn
import torchaudio
from transformers import BertConfig, BertModel, DistilBertConfig, DistilBertModel

from configs.base import Config
from torchvggish import vggish

def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


def build_DistilBert_encoder() -> nn.Module:
    """A function to build DistilBERT encoder"""
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=True, output_attentions=True)
    distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased", config=config)
    return distilbert


def build_miniBert_encoder() -> nn.Module:
    """A function to build miniBERT encoder"""    
    config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=8, output_hidden_states=True, output_attentions=True)
    minibert = BertModel.from_pretrained("bert-base-uncased",config=config)
    return minibert


def build_microBert_encoder() -> nn.Module:
    """A function to build microBERT encoder"""    
    config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=6, output_hidden_states=True, output_attentions=True)
    microbert = BertModel.from_pretrained("bert-base-uncased",config=config)
    return microbert


def build_nanoBert_encoder() -> nn.Module:
    """A function to build nanoBERT encoder"""    
    config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=4, output_hidden_states=True, output_attentions=True)
    nanobert = BertModel.from_pretrained("bert-base-uncased",config=config)
    return nanobert


def build_picoBert_encoder() -> nn.Module:
    """A function to build picoBERT encoder"""    
    config = BertConfig.from_pretrained("bert-base-uncased", num_hidden_layers=2, output_hidden_states=True, output_attentions=True)
    picobert = BertModel.from_pretrained("bert-base-uncased",config=config)
    return picobert


class VGGish(nn.Module):
    def __init__(self, postprocess):
        super(VGGish, self).__init__()
        self.vggish = vggish(postprocess)

    def forward(self, x):
        out = []
        for i in range(x.size(0)):
            out.append(self.vggish(x[i]))
        x = torch.stack(out, dim=0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)
        return x


def build_vggish_encoder(cfg: Config) -> nn.Module:
    """A function to build vggish encoder"""
    return VGGish(cfg.audio_postprocess)



def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "vggish": build_vggish_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
        "distilbert": build_DistilBert_encoder,
        "minibert": build_miniBert_encoder,
        "microbert": build_microBert_encoder,
        "nanobert": build_nanoBert_encoder,
        "picobert": build_picoBert_encoder
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()
