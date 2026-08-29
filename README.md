<p align="center">
  <h1 align="center">
    <strong>CheXGround: Anatomical Region Tokens for Grounded Longitudinal Chest X-ray Interpretation</strong>
  </h1>
</p>

<p align="center">
  Adonay Demewez Gebremedhin<sup>1</sup> · Wessam Shehieb<sup>1</sup> · Sara Alansari<sup>2</sup> · Mohamad Alansari<sup>3</sup> · Muzammal Naseer<sup>3,4</sup> · Sajid Javed<sup>3</sup> · Naoufel Werghi<sup>3</sup>
</p>

<p align="center">
  <small><sup>1</sup> Ajman University, UAE · <sup>2</sup> University of Birmingham, UK · <sup>3</sup> Khalifa University, UAE · <sup>4</sup> University of Western Australia, Australia</small>
</p>

<p align="center">
  <strong>✨ BMVC 2026 ✨</strong>
</p>

## Abstract

Recent radiology multi-modal language models have made substantial progress in chest X-ray report generation, visual question answering, and temporal reasoning. While longitudinal chest X-ray interpretation compares sequential examinations to describe change, visual grounding aims to connect clinical language with localized image evidence. Although longitudinal modeling and visual grounding have each advanced radiology language models, how localized visual evidence can support longitudinal interpretation remains under-explored. We introduce CheXGround, a region-grounded longitudinal chest X-ray language model that represents paired studies through corresponding anatomical regions. CheXGround extracts anatomical regions from current and prior radiographs, encodes them as temporally enhanced Region-of-Interest (ROI) tokens, and combines them with global temporal image context during generation. To connect these region tokens with clinical text, we propose Temporal Region–Phrase Alignment, a pretraining objective that aligns temporal anatomical representations with localized report phrases. We evaluate CheXGround on single-study and longitudinal Visual Question Answering (VQA), longitudinal findings generation, temporal grounded VQA, and anatomical grounding. Across these tasks, CheXGround improves clinical language quality, temporal reasoning, and localization accuracy over recent baselines. Our results suggest that organizing longitudinal evidence at the anatomical level is a strong representation for grounded radiology language modeling.

## Code

Coming soon.

## Citation

```bibtex
@inproceedings{gebremedhin2026chexground,
  title     = {{CheXGround}: Anatomical Region Tokens for Grounded Longitudinal Chest X-ray Interpretation},
  author    = {Adonay Demewez Gebremedhin and Wessam Shehieb and Sara Alansari and Mohamad Alansari and Muzammal Naseer and Sajid Javed and Naoufel Werghi},
  booktitle = {British Machine Vision Conference},
  year      = {2026}
}
```
