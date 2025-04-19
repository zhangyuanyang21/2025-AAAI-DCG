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

The Synthetic3D, CUB, HandWritten, LandUse-21, and Fashion datasets are placed in "data" folder. The others dataset could be downloaded from [cloud](). 

## 4.Usage

- an example for train a new model：

```bash
python run.py
```

## 5.Experiment Results

<img src="https://github.com/zhangyuanyang21/2025-AAAI-DCG/blob/main/figs/table1.png"  width="897"  />

## 6.Acknowledgments

Work&Code is inspired by [Completer](https://github.com/XLearning-SCU/2021-CVPR-Completer) ... 

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

If you have any problems, contact me via zhangyuanyang@seu.edu.cn
