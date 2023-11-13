# Domain Adaptation Transformer for Unsupervised Driving-Scene Segmentation in Adverse Conditions
Wenyu Liu, Song Wang, [Jianke Zhu](https://person.zju.edu.cn/jkzhu/645901.html), Xuansong Xie, [Lei Zhang](https://web.comp.polyu.edu.hk/cslzhang/)
## Requirements
* torch==1.7.1+cu110
* torchvision==0.8.2+cu110
* torchmetrics==0.9.1
* pytorch-lightning==1.5.10
* jsonargparse[signatures]==4.3.1
* pandas==1.4.2
* kornia==0.5.8
* opencv-python==4.6.0.66
* h5py==3.7.0

## Download the Pretrained Weights

Some pretrained weights are required for ACSegFormer. Save them to `./pretrained_models/`.

1. UAWarpC checkpoint, download it [Refign repository](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/626140/uawarpc_megadepth.ckpt).

2. ImageNet-pretrained MiT weights (`mit_b5.pth`), download them from the [SegFormer repository](https://github.com/NVlabs/SegFormer).

3. Cityscapes-pretrained SegFormer weights (`segformer.b5.1024x1024.city.160k.pth`), download them from the [SegFormer repository](https://github.com/NVlabs/SegFormer).

## Datasets and Models
**Cityscapes**:  [Cityscape](https://www.cityscapes-dataset.com/) 
**NightCity**:  [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html/) 
**ACDC**:  [ACDC](https://acdc.vision.ee.ethz.ch/) 
**Dark-Zurich**: [Dark-Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)  

**Models**: [[Google Drive](https://drive.google.com/drive/folders/1s3xsAyCDEwn1parY3YM15u5ntjvTG9kR)]

## Refign Testing

To evaluate ACSegFormer e.g. on the ACDC validation set, use the following command:

```bash
python start.py test --config configs/cityscapes_acdc/refign_hrda_star.yaml --ckpt_path /path/to/trained/model --trainer.gpus 1
```
To get test set scores for ACDC and DarkZurich, predictions are evaluated on the respective evaluation servers: [ACDC](https://acdc.vision.ee.ethz.ch/submit) and [DarkZurich](https://codalab.lisn.upsaclay.fr/competitions/3783).
To create and save test predictions for e.g. ACDC, use this command:
```bash
python start.py predict --config configs/cityscapes_acdc/refign_hrda_star.yaml --ckpt_path /path/to/trained/model --trainer.gpus 1
```
## Acknowledgments
The code is based on Refign, HRDA.
