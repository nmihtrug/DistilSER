
<h1 align="center">
  DistilSER
  <br>
</h1>

<h4 align="center">Official code repository for paper "Enhancing Speech Emotion Recognition through Knowledge Distillation". Paper submitted to <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?reload=true&punumber=5165369">IEEE Transactions on Affective Computing (2024)</a> </h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/TrungAzieL/DistilSER?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/TrungAzieL/DistilSER?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/TrungAzieL/DistilSER?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> Speech Emotion Recognition (SER) is becoming more and more crucial in applications including healthcare, entertainment, customer service, and human-computer interaction. However, modern SER models are often heavy and slow, making them unsuitable for real-time applications. To address this problem, we propose a technique that creates a lightweight student model through knowledge distillation. We leverage the solid performance of the 3M-SER model as the teacher in this method. While our student model keeps VGGish for audio embedding, it replaces BERT with smaller variants, such as MiniBERT, MicroBERT, NanoBERT, or PicoBERT. This approach is appropriate for real-time SER applications since it produces lighter and more effective outcomes while preserving or even enhancing performance in comparison to the traditional 3M-SER.

## How To Use
- Clone this repository 
```bash
git clone https://github.com/TrungAzieL/DistilSER.git 
cd DistilSER
```
- Create a conda environment and install requirements
```bash
conda create -n DistilSER python=3.10 -y
conda activate DistilSER
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). 
- Preprocess data or you can download our preprocessed dataset [here](https://github.com/TrungAzieL/DistilSER/releases) (this only include path to sample in dataset).

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/DistilSER_bert_vggish.py
```

- The visualization of our figure in paper can be found in [notebook](./src/visualization/metrics.ipynb).

- You can also find our pre-trained models in the [release](https://github.com/TrungAzieL/DistilSER/releases).

## Citation
```bibtex

```
## References

[1] Phuong-Nam Tran, 3M-SER: Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings (INISCOM), 2023. Available https://github.com/namphuongtran9196/3m-ser.git

[2] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@TrungAzieL](https://github.com/TrungAzieL) &nbsp;&middot;&nbsp;