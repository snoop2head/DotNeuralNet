# DotNeuralNet

**Light-weight Neural Network for Optical Braille Recognition in the wild & on the book.**

- Classified multi label one-hot encoded labels for raised dots.
- Pseudo-labeled Natural Scene Braille symbols.
- Trained single stage object detection YOLO models for Braille symbols.

### Repository Structure

### Result

- Inferenced result of yolov8-m model on validation subset.
  ![yolov8 img](./assets/result_yolov8.png)
- Inferenced result of yolov5-m model on validation subset.
  ![yolov5 img](./assets/result_yolov5.png)

### Logs

- Train / Validation log of yolov8-m model
  ![yolov8 log](./assets/log_yolov8_long.png)
- Train / Validation log of yolov5-m model available at [🔗 WandB](https://wandb.ai/snoop2head/YOLOv5/runs/mqvmh4nc)
  ![yolov8 log](./assets/log_yolov5.png)

### References

```
@article{Li2018DSBIDouble-SidedBraille,
    title   = {DSBI: Double-Sided Braille Image Dataset and Algorithm Evaluation for Braille Dots Detection},
    author  = {Renqiang Li, Hong Liu, Xiangdong Wan, Yueliang Qian},
    journal = {ArXiv},
    year    = {2018},
    volume  = {abs/1811.1089}
}
```

```
@article{Ovodov2021OpticalBrailleRecog,
    title   = {Optical Braille Recognition Using Object Detection CNN},
    author  = {Ilya G. Ovodov},
    journal = {2021 IEEE/CVF International Conference on Computer Vision Workshops},
    year    = {2021},
    pages   = {1741-1748}
}
```

```
@article{lu2022AnchorFreeBrailleCharac
    title   = {Anchor-Free Braille Character Detection Based on Edge Feature in Natural Scene Images},
    author  = {Liqiong Lu, Dong Wu, Jianfang Xiong, Zhou Liang and Faliang Huang},
    journal = {Computational Intelligence and Neuroscience},
    year    = {2022},
    url     = {https://www.hindawi.com/journals/cin/2022/7201775}
}
```
