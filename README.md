
<h1 align="center">
  DistilSER
  <br>
</h1>

<h4 align="center">Official code repository for paper "Enhancing Speech Emotion Recognition through Knowledge Distillation". Paper submitted to <a href="https://ictc.org">The 15th International Conference on ICT Convergence (ICTC 2024)</a> </h4>

<p align="center">
  <a href=""><img src="https://img.shields.io/github/stars/nmihtrug/DistilSER?" alt="stars"></a>
  <a href=""><img src="https://img.shields.io/github/forks/nmihtrug/DistilSER?" alt="forks"></a>
  <a href=""><img src="https://img.shields.io/github/license/nmihtrug/DistilSER?" alt="license"></a>
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
> The importance of Speech Emotion Recognition (SER) is growing in diverse applications, which has resulted in the development of multiple methodologies and models to improve SER performance. Nevertheless, modern SER models can require significant processing resources and exhibit poor performance, making them unsuitable for real-time applications.  To address this, we propose a method using knowledge distillation to generate compact, efficient student models derived from the 3M-SER architecture. Our approach replaces BERT with smaller variants like MicroBERT, NanoBERT, or PicoBERT for text embedding, while retaining VGGish for audio embedding. Our approach reduces model size by up to 44.9\% and improves inference time by up to 40.2\%. Experiments on the IEMOCAP dataset demonstrate that our proposed student models, when trained with knowledge distillation, can achieve comparable or superior accuracy to the teacher model. These results underscore the effectiveness of knowledge distillation in creating efficient and accurate SER models suitable for resource-constrained environments and real-time applications. Our work contributes to the ongoing effort to make advanced SER technology more accessible and deployable in practical settings.

## How To Use
- Clone this repository 
```bash
git clone https://github.com/nmihtrug/DistilSER.git 
cd DistilSER
```
- Create a conda environment and install requirements
```bash
conda create -n DistilSER python=3.10 -y
conda activate DistilSER
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). 
- Preprocess data or you can download our preprocessed dataset [here](https://github.com/nmihtrug/DistilSER/releases) (this only include path to sample in dataset).

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/bert_vggish.py
```

- The visualization of our figure in paper can be found in [notebook](./src/visualization/metrics.ipynb).

- You can also find our pre-trained models in the [release](https://github.com/nmihtrug/DistilSER/releases).

## Citation
```bibtex

```
## References

[1] Phuong-Nam Tran, 3M-SER: Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings (INISCOM), 2023. Available https://github.com/namphuongtran9196/3m-ser.git

[2] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@nmihtrug](https://github.com/nmihtrug) &nbsp;&middot;&nbsp;