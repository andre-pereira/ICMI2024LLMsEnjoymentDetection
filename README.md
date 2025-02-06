# Multimodal User Enjoyment Detection in Human-Robot Conversation

## Overview

This repository accompanies the paper:

**Pereira, A., Marcinek, L., Miniota, J., Thunberg, S., Lagerstedt, E., Gustafson, J., Skantze, G., & Irfan, B. (2024).** *Multimodal User Enjoyment Detection in Human-Robot Conversation: The Power of Large Language Models*. In International Conference on Multimodal Interaction (ICMI '24), November 04–08, 2024, San Jose, Costa Rica. [ACM DOI: 10.1145/3678957.3685729](https://dl.acm.org/doi/abs/10.1145/3678957.3685729)

## Repository Contents

This repository contains the source code used in the paper to evaluate Large Language Models (LLMs) and multimodal approaches for detecting user enjoyment in Human-Robot Interaction (HRI). Specifically, the repository includes:

- **Prompting Code:** Scripts used to prompt OpenAI's GPT-3.5/GPT-4 and Google's Gemini for user enjoyment annotation.
- **Supervised Learning Models:** Code for training Long Short-Term Memory (LSTM) and XGBoost classifiers on multimodal datasets.
- **Evaluation Scripts:** Code for computing evaluation metrics such as accuracy, Mean Absolute Error (MAE), F1-score, and Balanced Accuracy (BA).
- **Raw Multimodal Data:** The dataset includes extracted audio, video, and turn-taking data from interactions. This can be useful for benchmarking different multimodal machine learning methods.
- **Model Outputs:** The `OutputOfModelsUsed/` directory contains the outputs from the various models described and evaluated in the paper.

## Dataset

Due to anonymity concerns, the dataset used in the paper has been anonymized and can be accessed at [Zenodo](https://zenodo.org/records/12588810). We are not sharing the actual logs of prompts used step by step, as they contained identifiable information. Similarly, the reasoning outputs have been omitted for anonymity preservation. Additionally, we cannot share the original videos, as we do not have permission to do so. However, in this repository, we also provide the raw outputs from the feature extraction tools used in the paper for benchmarking multimodal models. Please contact us if you would like to establish a deeper collaboration with us regarding this dataset or work.

## Code Adaptation

The code in this repository does not currently run as-is. It must be adapted to work with the new anonymized dataset structure. Users will need to refactor dataset handling components before executing any scripts.

## Citation

If you use this repository, please cite our paper:

```bibtex
@inproceedings{pereira2024enjoyment,
  author = {Pereira, Andre and Marcinek, Lubos and Miniota, Jura and Thunberg, Sofa and Lagerstedt, Erik and Gustafson, Joakim and Skantze, Gabriel and Irfan, Bahar},
  title = {Multimodal User Enjoyment Detection in Human-Robot Conversation: The Power of Large Language Models},
  booktitle = {Proceedings of the International Conference on Multimodal Interaction},
  year = {2024},
  publisher = {ACM},
  doi = {10.1145/3678957.3685729}
}
```

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License.

For any questions, please contact the authors via their institutional emails.

