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

