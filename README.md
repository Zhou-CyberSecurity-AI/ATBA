<p align="center">
    <img src="docs/images/logo.svg" width = "400"  alt="ATBA Attack" align=center />
</p>
<p align="center">
  <a target="_blank">
    <img src="https://github.com/thunlp/OpenAttack/workflows/Test/badge.svg?branch=master" alt="Github Runner Covergae Status">
  </a>
  <a href="" target="_blank">
    <img src="https://readthedocs.org/projects/openattack/badge/?version=latest" alt="ReadTheDoc Status">
  </a>
  <a  href="https://pypi.org/project/OpenAttack/"  target="_blank">
    <img src="https://img.shields.io/pypi/v/OpenAttack?label=pypi" alt="PyPI version">
  </a>
  <a  href="https://github.com/thunlp/OpenAttack/releases"  target="_blank">
    <img src="https://img.shields.io/github/v/release/thunlp/OpenAttack" alt="GitHub release (latest by date)">  
  </a>
  <a target="_blank">
    <img alt="GitHub" src="https://img.shields.io/github/license/thunlp/OpenAttack">
  </a>
   <a target="_blank">
    <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs are Welcome">
  </a>
<br><br>
  <a href="https://openattack.readthedocs.io/" target="_blank">Documentation</a> • <a href="#features--uses">Features & Uses</a> • <a href="#usage-examples">Usage Examples</a> • <a href="#attack-models">Attack Models</a> • <a href="#toolkit-design">Toolkit Design</a> 
<br>
</p>

<p style="text-align: center;">
  <b>ATBA: Transferring Backdoors between Large Language Models by Knowledge Distillation </b>
</p>

<div align="center">
<img src="pipeline.png" alt="Centered Image" style="width:500px;"/>
</div>

**Contribution:**

1. We propose ATBA, the first adaptive and transferable backdoor attack for LLMs, which aims to reveal the vulnerability of LLMs when using knowledge distillation.

2. We design a target trigger generation module that leverages cosine similarity distribution to filter out indicative triggers from the original vocabulary tables of the teacher LLMs. This approach not only effectively realizes implicit backdoor transferable but also reduces search complexity.

3. We introduce an adaptive trigger optimization module based on KD simulation and dynamic greedy searching, which overcomes textual discretization and is more robust than traditional triggers.

4. Extensive experiments show that ATBA is highly transferable and successfully activates against student models with different architectures on five popular tasks. 

**How to Running ATBA**

*1. Environment*
```shell
pip install -r reuirement.txt
```

*2. Download Dataset from HuggingFace*
```python
from datasets improt load_dataset
dataset = load_dataset("dataset path")
dataset.save_to_disk("./dataset/")
```


*3. Download Models from HuggingFace*
```python
model.save_pretrained("/home/models/")
```

*4. Warmup*

Warm up the model using the **warmup.ipynb** script in the ATO module

*5. TTG*

Modify the model and dataset paths under run/TTG_xxx.sh and run the corresponding script to obtain the target trigger word candidates.
```shell
bash ./run/TTG_xxx.sh
```
*6. ATO*

Modify the model and dataset paths under run/ATO_xxx.sh and run the corresponding script to get the optimal trigger word.

```shell
bash ./run/ATO_xxx.sh
```

*7. Evaluation*

Modify the model and dataset paths under run/KD_xxx.sh and run the corresponding script to evaluate the backdoor transfer capability of the teacher model on the three student models.

```shell
bash ./run/KD_xxx.sh
```

## Attack Models


## Citation

Please cite our [paper](https://arxiv.org/pdf/2408.09878) if you use this toolkit:

```
@article{cheng2024transferring,
  title={Transferring backdoors between large language models by knowledge distillation},
  author={Cheng, Pengzhou and Wu, Zongru and Ju, Tianjie and Du, Wei and Liu, Zhuosheng Zhang Gongshen},
  journal={arXiv preprint arXiv:2408.09878},
  year={2024}
}
```

## Contributors
We thank all the contributors to this project. And more contributions are very welcome.

![Contributors](https://contrib.rocks/image?repo=Zhou-CyberSecurity-AI/ATBA)

