# MTPNet: Enhancing Multimodal Fusion with Masked Token Pretraining

### Task: Multimodal Vision and Vision-Language

MTPNet introduces a general-purpose multimodal foundation model designed to effectively address a wide range of vision and vision-language tasks, such as instance segmentation, object detection, and visual question answering. The primary goal is to create a scalable model that delivers state-of-the-art results across these different domains, demonstrating its ability to seamlessly adapt to various visual and multimodal challenges. By using a unified architecture, MTPNet reduces the need for task-specific models, allowing it to generalize well across multiple domains and provide consistent, high-quality results. The model aims to streamline the development of vision-based applications by serving as a flexible solution for numerous tasks.

### Overview

MTPNet employs a general-purpose multimodal pretraining framework that integrates both visual and language data, aiming to achieve cutting-edge performance across a variety of downstream tasks. Traditional vision-language models often require modifying the model architecture for specific tasks or use separate parameters for each modality, limiting their efficiency and scalability. To overcome these limitations, MTPNet adopts a shared multi-head attention-based Transformer architecture that treats images similarly to text by using a Masked Token Pretraining approach. This unified architecture supports deep fusion between modalities and enables both modality-specific encoding and cross-modal integration.



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

