
<h1 align="center">
  DistilSER
  <br>
</h1>

<h4 align="center">Official code repository for the paper "Enhancing Speech Emotion Recognition through Knowledge Distillation". Paper accepted at <a href="https://ictc.org">The 15th International Conference on ICT Convergence (ICTC 2024)</a> </h4>

<p align="center">
  <a href=""><img src="https://img.shields.io/github/stars/nmihtrug/DistilSER?" alt="stars"></a>
  <a href=""><img src="https://img.shields.io/github/forks/nmihtrug/DistilSER?" alt="forks"></a>
  <a href=""><img src="https://img.shields.io/github/license/nmihtrug/DistilSER?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> The importance of Speech Emotion Recognition (SER) is growing across diverse applications, which has resulted in the development of multiple methodologies and models to improve SER performance. Nevertheless, some modern SER models require significant processing resources and exhibit poor performance, making them unsuitable for real-time applications. To address this, we propose a novel approach that leverages Knowledge Distillation (KD) to create lightweight student models derived from the 3M-SER architecture. Our method focuses on compressing the text embedding component by replacing BERT<sub>BASE</sub> with smaller variants while maintaining VGGish for audio embedding. Experiments conducted on the IEMOCAP dataset demonstrate that our student model, which reduces model size by up to 44.9%, achieves performance remarkably close to that of the teacher model while improving inference time by up to 40.2% when trained with KD. These results underscore the effectiveness of KD in creating efficient and accurate SER models suitable for resource-constrained environments and real-time applications. Our work contributes to the ongoing effort to make advanced SER technology more accessible and deployable in practical settings.
## How To Use
### Clone this repository 
```bash
git clone https://github.com/nmihtrug/DistilSER.git 
cd DistilSER
```
### Create a conda environment and install requirements
```bash
conda create -n DistilSER python=3.10 -y
conda activate DistilSER
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Download dataset and preprocess 

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). 
- Preprocess data or you can download our preprocessed dataset [here](https://github.com/nmihtrug/DistilSER/releases) (this only include path to sample in dataset).

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

### Training model
- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

- Train teacher model
```bash
cd scripts && python train.py -cfg ../src/configs/bert_vggish.py # Train teacher model
```

- Train student model
```bash
cd scripts && python train_distillation.py -stu_cfg ../src/configs/nanobert_vggish.py # Train student model
```

### Evaluation and Visualization
```bash
cd scripts && python eval.py -ckpt checkpoints_latest/student_kd/_4M_SER_nanobert_vggish/20240625-045938/weights/best_acc/checkpoint_58_259782.pt
```

- The visualization of our figure in paper can be found in [notebook](./src/visualization/metrics.ipynb).

- You can also find our pre-trained models in the [release](https://github.com/nmihtrug/DistilSER/releases).

## Citation
```bibtex
  Update soon
```
## References

[1] Phuong-Nam Tran, 3M-SER: Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings (INISCOM), 2023. Available https://github.com/tpnam0901/3m-ser.git

[2] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@nmihtrug](https://github.com/nmihtrug) &nbsp;&middot;&nbsp;
