# TICoA
This repo is for the paper "Text-Image Co-Alignment for Weakly Supervised Polyp Segmentation"
## Overall Framework

![TICoA Framework](assets/model.png)

## Environment Setup

### Requirements
- Python 3.10.13
- PyTorch 2.1.1
- CUDA 11.8

### Create Conda Environment
```bash
conda create -n TICoA python=3.10.13
conda activate TICoA

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

```markdown
### Hardware
- NVIDIA GPU with CUDA 11.8 support is recommended.
