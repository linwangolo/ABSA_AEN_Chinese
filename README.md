# Opinion_AEN

This is a package for **chinese target-specific opinion retrieval**. 
Retrained AEN-BERT on CCF2012 datasets.

## Pretrained model
You need to put the model file into `opinion_aen/state_dict/`, and the pretrained model can be downloaded from [here](https://drive.google.com/file/d/1aOe6jk9ODwSesRC3TLiozp_71pfJWxvc/view?usp=sharing).
The pretrained model is trained with **weibo corpus in Simplified Chinese** and the accruracy achieves **90.48**.

## Installation
```shell
git clone https://github.com/linwangolo/ABSA_AEN_Chinese.git
pip install opinion_aen/
```

## Usage
```python
import opinion_aen

model_path = '~/opinion_aen/state_dict/aen_bert_CCF_val_acc0.9048'
model = opinion_aen.model(model_path)
inputs = opinion_aen.Input(data).data  # do some input preprocessing
prob, polar = model.predict(inputs)
```


###### [Reference] Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1903.09314 (2019)[Github](https://github.com/songyouwei/ABSA-PyTorch)

