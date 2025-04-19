# 2025-AAAI-DCG
Incomplete Multi-view Clustering via Diffusion Contrastive Generation

## Incomplete Multi-view Clustering via Diffusion Contrastive Generation
> **Authors:**
Yuanyang Zhang#, Yijie Lin#, Weiqing Yan, Li Yao†, Xinhang Wan, Guangyuan Li, Chao Zhang, Guanzhou Ke, Jie Xu

This repo contains the code and data of our AAAI'2025 paper [Incomplete Multi-view Clustering via Diffusion Contrastive Generation
](https://ojs.aaai.org/index.php/AAAI/article/view/34424).

## 1. Framework

<img src="https://github.com/zhangyuanyang21/2025-AAAI-DCG/blob/main/figs/framework.png"  width="897" height="317" />

The overall framework. Without loss of generality, we take bi-view data as a showcase to demonstrate the overall framework of our proposed
DCG. As shown, our method is mainly divided into four modules: (1) View-specifc Feature Learning; (2) Diffusion Contrastive
Generation; (3) Instance-level Interactive Learning; (4) Category-level Interactive Learning. Note that, E: Encoder; D: Decoder;
DM: Diffusion Model; CL: Contrastive Learning.

## 2.Requirements

pytorch==1.12.1

numpy>=1.21.6

scikit-learn>=1.0.2

## 3.Datasets

The Synthetic3d, Prokaryotic, and MNIST-USPS datasets are placed in "data" folder. The others dataset could be downloaded from [cloud](https://pan.baidu.com/s/1XNWW8UqTcPMkw9NpiKqvOQ). key: data

## 4.Usage

- Before run, please carefully read  ''Obtain-S.docx'', and refer to the steps inside it to modify the code in order to obtain S.

- an example for train a new model：

```bash
python train.py
```
- an example  for test the trained model:
  
```bash
python test.py
```

- You can get the following output:

```bash
Epoch 290 Loss:15.420288
Epoch 291 Loss:15.431067
Epoch 292 Loss:15.417261
Epoch 293 Loss:15.436375
Epoch 294 Loss:15.398655
Epoch 295 Loss:15.406467
Epoch 296 Loss:15.413018
Epoch 297 Loss:15.419146
Epoch 298 Loss:15.419894
Epoch 299 Loss:15.389602
Epoch 300 Loss:15.377309
---------train over---------
Clustering results:
ACC = 0.9700 NMI = 0.8713 PUR=0.9700 ARI = 0.9126
Saving model...
```

## 5.Experiment Results

<img src="https://github.com/zhangyuanyang21/2025-AAAI-DCG/blob/main/figs/table1.png"  width="897"  />

## 6.Acknowledgments

Work&Code is inspired by [MFLVC](https://github.com/SubmissionsIn/MFLVC) ... 

## 7.Citation

If you find our work useful in your research, please consider citing:

```latex
@inproceedings{zhang2025incomplete,
  title={Incomplete Multi-view Clustering via Diffusion Contrastive Generation},
  author={Zhang, Yuanyang and Lin, Yijie and Yan, Weiqing and Yao, Li and Wan, Xinhang and Li, Guangyuan and Zhang, Chao and Ke, Guanzhou and Xu, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={22650--22658},
  year={2025}
}
```

If you have any problems, contact me via zhangyuanyang@seu.edu.cn.
