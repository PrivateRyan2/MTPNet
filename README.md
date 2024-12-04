# MTPNet: Enhancing Multimodal Fusion with Masked Token Pretraining

### Setup

1. Install the required libraries

```python
pip install -r requirements.txt
```

2. Download the datasets 
   - [COCO 2017](https://cocodataset.org/#download)
   - [VQAv2](https://visualqa.org/download.html)

### Train

```
python  train.py \
        --input_size 480 --task vqav2 --batch_size 32 --layer_decay 1.0 --lr 1e-4 --epochs 100 \ 
        --warmup_epochs 5 --data_path <data path> --output_dir <model save path> \
        --log_dir <log path> --weight_decay 0.01 --seed 14 \
        --save_ckpt_freq 5 --task_head_lr_weight 20 --opt_betas 0.9 0.98
```

### Evaluation

```
python  train.py \
        --input_size 480 --task vqav2 --batch_size 32 --data_path <data path> \
        --output_dir <prediction output path> \
        --eval
```



### Experiment Results

#### Instance Segmentation on COCO benchmark

|    Model    | mask AP  |
| :---------: | :------: |
|   Swin-L    |   51.1   |
| ViT-Adapter |   52.1   |
|  Mask DINO  |   52.5   |
| **MTPNet**  | **54.3** |

#### Object Detection on COCO benchmark

|    Model    | box mAP  |
| :---------: | :------: |
|   Swin-L    |   60.6   |
| ViT-Adapter |   60.9   |
|    GLIP     |   61.5   |
| **MTPNet**  | **62.8** |

#### Visual Quesion Answering on VQAv2 benchmark

|   Model    | vqa-score |
| :--------: | :-------: |
|   VinVL    |   76.52   |
|    BLIP    |   78.25   |
|    VAST    |   80.16   |
| **MTPNet** | **83.19** |

