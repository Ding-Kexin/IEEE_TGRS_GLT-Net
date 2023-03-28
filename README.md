# GLT-Net
The repository contains the implementations for "**Global-local Transformer Network for HSI and LiDAR Data Joint Classification**". You can find [the PDF of this paper](https://ieeexplore.ieee.org/abstract/document/9926173).
![GLT-Net](https://github.com/Ding-Kexin/GLT-Net/blob/main/figure/GLT-Net.jpg)
****
# Datasets
- [MUUFL](https://github.com/GatorSense/MUUFLGulfport/)
- [Trento](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)
****
# Train GLT-Net
``` 
python demo.py
``` 
****
# Results
| Dataset | OA (%) | AA (%) | Kappa (%) |
| :----: | :----: | :----: | :----: |
| MUUFL  | 85.23 | 84.67 | 80.96 |
| Trento  | 99.27 | 98.63 | 99.03 |
| Houston  | 95.32 | 96.01 | 94.95 |
****
# Citation
If you find this paper useful, please cite:
``` 
@ARTICLE{9926173,
  author={Ding, Kexing and Lu, Ting and Fu, Wei and Li, Shutao and Ma, Fuyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Global–Local Transformer Network for HSI and LiDAR Data Joint Classification}, 
  year={2022},
  volume={60},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2022.3216319}}
}
```
****
# Contact
Kexin Ding: [dingkexin@hnu.edu.cn](dingkexin@hnu.edu.cn)
