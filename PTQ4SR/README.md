# Contents

- [PTQ4SR Description](#PTQ4SR-description)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Evaluation Process](#evaluation-process)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Training Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)
- [Reference](#reference)

# [PTQ4SR Description](#contents)

Model quantization is a crucial step for deploying super esolution (SR) networks on mobile devices. However, existing works focus on quantization-aware training, which requires complete dataset and expensive computational overhead. In this paper, we study post-training quantization (PTQ) for image super resolution using only a few unlabeled calibration images. As the SR model aims to maintain the texture and color information of input images, the distribution of activations are long-tailed, asymmetric and highly dynamic compared with classification models. To this end, we introduce the density-based dual clipping to cut off the outliers based on analyzing the asymmetric bounds of activations. Moreover, we present a novel pixel aware calibration method with the supervision of the full-precision model to accommodate the highly dynamic range of different samples. Extensive experiments demonstrate that the proposed method significantly outperforms existing PTQ algorithms on various models and datasets. For instance, we get a 2.091 dB increase on Urban100 benchmark when quantizing EDSR×4 to 4-bit with 100 unlabeled images.

[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf): Zhijun Tu, Jie Hu, Hanting Chen, Yunhe Wang. Toward Accurate Post-Training Quantization for Image Super Resolution. Accepted by CVPR 2023.

# [Dataset](#contents)

- Dataset used: [DIV-2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
    - train data: starting from 800 high definition high resolution images we obtain corresponding low resolution images and provide both high and low resolution images for 2, 3, and 4 downscaling factors
    - validation data: 100 high definition high resolution images are used for genereting low resolution corresponding images, the low res are provided from the beginning of the challenge and are meant for the participants to get online feedback from the validation server; the high resolution images will be released when the final phase of the challenge starts.
    - test data: 100 diverse images are used to generate low resolution corresponding images; the participants will receive the low resolution images when the final evaluation phase starts and the results will be announced after the challenge is over and the winners are decided.

# [Features](#contents)

## [Mixed Precision(Ascend)](#contents)

The mixed precision training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware（Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend、GPU or CPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorial/en/r0.5/index.html)
    - [MindSpore API](https://www.mindspore.cn/api/en/0.1.0-alpha/index.html)

# [Script description](#contents)

## [Script and sample code](#contents)

```text
├── PTQ4SR
  ├── README.md       # readme
  ├── model_utils
  │   ├──config.py
  │   ├──device_adapter.py
  │   ├──moxing_adapter.py
  ├── src
  │   ├──loss.py      # loss and metrics
  │   ├──dataset.py   # creating dataset
  │   ├──QuantEDSR.py # EDSR architecture with quantization noes
  │   ├──quantlib.py  # quantizer
  ├── eval.py         # evaluation script
  ├── mindspore_hub_conf.py
```

## [Evaluation Process](#contents)

### Usage

After installing MindSpore via the official website, you can start evaluation as follows:

### Launch

```bash
# infer example
# 运行评估示例(EDSR(x2) in the paper)
python eval.py --config_path DIV2K_config.yaml --scale 2 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x2 model path] > train.log 2>&1 &
# 运行评估示例(EDSR(x3) in the paper)
python eval.py --config_path DIV2K_config.yaml --scale 3 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x3 model path] > train.log 2>&1 &
# 运行评估示例(EDSR(x4) in the paper)
python eval.py --config_path DIV2K_config.yaml --scale 4 --data_path [DIV2K path] --output_path [path to save sr] --pre_trained [pre-trained EDSR_x4 model path] > train.log 2>&1 &
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

#### AdaBin on CIFAR-10

| Parameters                 |                                        |
| -------------------------- | -------------------------------------- |
| Model Version              | EDSR         |
| uploaded Date              | 12/13/2023 (month/day/year)  ；                     |
|  Device                    | GPU |
| MindSpore Version          | 1.8.0                                                    |
| Dataset                    | DIV2K                                             |
| Input size   | 48x48                                       |
| Training Time (min) | 10 |
| Training Time per step (s) | 0.18 |

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside "create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).

# [Reference](#reference)

[EDSR](https://gitee.com/mindspore/models/tree/master/official/cv/EDSR)

