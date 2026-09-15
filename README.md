<p align="center">
  <img src="assets/logo.jpeg" alt="CheXGround logo" width="72">
</p>

<p align="center">
  <h1 align="center">
    <strong>CheXGround: Anatomical Region Tokens for Grounded Longitudinal Chest X-ray Interpretation</strong>
  </h1>
</p>

<p align="center">
  <a href="https://adonaydem.github.io/">Adonay Demewez Gebremedhin</a><sup>1</sup> · <a href="https://scholar.google.com/citations?user=5oaOk_YAAAAJ&amp;hl=en">Wessam Shehieb</a><sup>1</sup> · <a href="https://scholar.google.com/citations?user=h1w_MmAAAAAJ">Sara Alansari</a><sup>2</sup> · <a href="https://scholar.google.com/citations?user=dLQ1jLkAAAAJ">Mohamad Alansari</a><sup>3</sup> · <a href="https://scholar.google.com/citations?user=tM9xKA8AAAAJ">Muzammal Naseer</a><sup>3,4</sup> · <a href="https://scholar.google.com/citations?user=6qvbEhUAAAAJ">Sajid Javed</a><sup>3</sup> · <a href="https://scholar.google.com/citations?user=G_2Xpm0AAAAJ">Naoufel Werghi</a><sup>3</sup>
</p>

<p align="center">
  <small><sup>1</sup> Ajman University, UAE · <sup>2</sup> University of Birmingham, UK · <sup>3</sup> Khalifa University, UAE · <sup>4</sup> University of Western Australia, Australia</small>
</p>

<p align="center">
  <strong>✨ BMVC 2026 ✨</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.30758"><img src="https://img.shields.io/badge/Paper-2608.30758-b31b1b?logo=arxiv&amp;logoColor=white" alt="Paper"></a>
  <a href="https://adonaydem.github.io/chexground-website/"><img src="https://img.shields.io/badge/Project-Page-2ea44f" alt="Project Page"></a>
  <a href="https://huggingface.co/adonaydem/CheXGround"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face model"></a>
</p>

## Summary

We introduce CheXGround, a region-grounded longitudinal chest X-ray language model that represents paired studies through corresponding anatomical regions. CheXGround extracts anatomical regions from current and prior radiographs, encodes them as temporally enhanced Region-of-Interest (ROI) tokens, and combines them with global temporal image context during generation. To connect these region tokens with clinical language representations, we propose Temporal Region–Phrase Alignment, a pretraining objective that aligns temporal anatomical representations with localized report phrases.

<p align="center">
  <a href="assets/task-overview.png">
    <img src="assets/task-overview.png" alt="Overview of the grounded radiology tasks" width="75%">
  </a>
</p>

<p align="center">
  <a href="assets/method-region-phrase.png">
    <img src="assets/method-region-phrase.png" alt="Temporal anatomical ROI encoding and region-phrase alignment" width="49%">
  </a>
  <a href="assets/method-architecture.png">
    <img src="assets/method-architecture.png" alt="CheXGround architecture" width="49%">
  </a>
</p>

## To-do

- [x] Code released
- [x] Model released
- [ ] Data protocol

## Installation

The setup was ran on Python 3.10, PyTorch 2.4.0, and CUDA 12.1.


```bash
git clone https://github.com/adonaydem/chexground.git
cd chexground
conda env create -f environment.yml
conda activate chexground

python -m pip install "setuptools==80.10.2" wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation -e ".[train,eval]"
python -m pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

Build the bundled MMCV CUDA operators for your GPU:

```bash
bash scripts/build_mmcv.sh
python -m pip check
python -m chexground.eval.run_chexground --help
```

## Model

Download the full [CheXGround model](https://huggingface.co/adonaydem/CheXGround):

```bash
hf download adonaydem/CheXGround --local-dir checkpoints/chexground
```

## Training

Set the data paths in `chexground/data/configs/` and replace the checkpoint paths
below. 

### Stage 0: Anatomical region detection

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/det_pretrain.sh \
  output/s0 chexground/data/configs/det_pretrain.py
```

### Stage 1: Temporal region–phrase alignment

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/temporal_grounding_pretrain.sh \
  /path/to/s0-checkpoint output/s1 /path/to/region-annotations \
  chexground/data/configs/temporal_grounding_mimic.py
```

### Stage 2: Vision-language pretraining
Our base VLM model is Meditron-7B finetuned by Libra: https://huggingface.co/X-iZhang/libra-v1.0-7b . Projectors are freshly initialized.
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/vl_pretrain.sh \
  /path/to/pretrained-vlm /path/to/s1-checkpoint \
  output/s2 chexground/data/configs/chexground_pretrain.py
```

### Stage 3: Instruction tuning

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/vl_finetune.sh \
  /path/to/s2-checkpoint output/s3 \
  chexground/data/configs/chexground_finetune.py
```

## Inference

Save input data in `data.jsonl`. List images chronologically, with the current
image last. A single current image is also supported.

```jsonl
{"id": "example", "image_refs": ["/path/to/prior.png", "/path/to/current.png"], "prompt": "Describe the interval changes."}
```

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_chexground.sh --model \
  checkpoints/chexground data.jsonl predictions.jsonl
```

For batched generation with predicted anatomical regions and boxes:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_chexground_batched.sh --model \
  checkpoints/chexground data.jsonl predictions.jsonl --batch-size 2
```


## Acknowledgements

CheXGround is built upon [Libra](https://github.com/X-iZhang/Libra) and [Groma](https://github.com/FoundationVision/Groma), we thank the authors for their contributions.

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).
Third-party components and pretrained models remain subject to their respective
licenses.

## If you find this work useful, please cite us!

```bibtex
@misc{gebremedhin2026chexgroundanatomicalregiontokens,
  title={CheXGround: Anatomical Region Tokens for Grounded Longitudinal Chest X-ray Interpretation},
  author={Adonay Demewez Gebremedhin and Wessam Shehieb and Sara Alansari and Mohamad Alansari and Muzammal Naseer and Sajid Javed and Naoufel Werghi},
  year={2026},
  eprint={2608.30758},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.30758},
}
```

## Intended use cases

CheXGround is intended for research and educational use, including:

- Studying grounded report generation and visual question answering for chest X-rays.
- Evaluating anatomical localization and reasoning about changes between prior and current studies.
- Developing and comparing multimodal learning methods on appropriately authorized research data.

**CheXGround is not meant to be used for clinical practice.**


## Disclaimer

CheXGround is an experimental research system. Its outputs are not medical
advice and must not be relied on for patient care. Although CheXGround shows competitive performance, subtle inaccuracies in anatomical localization and descriptions may still occur. 
