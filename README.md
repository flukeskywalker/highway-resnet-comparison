# Weighted Skip Connections are Not Harmful for Deep Nets

Comparison of 110-layer ResNets and HighwayNets on CIFAR-10, following [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027).

TL;DR:
> The paper Identity Mappings in Deep Residual Networks has design mistakes leading to incorrect conclusions about training deep networks with gated skip connections. You should try gated/weighted skip connections yourself and see if they improve results on your problems.

See this [accompanying blog post]() for details.


Requirement: The only requirements are pytorch and torchvision (for CIFAR10). Original results used pytorch 2.5.1.

```commandline
pip install torch torchvision
```

To train the ResNet110 baseline:
```commandline
python -u trainer.py --arch=resnet110 --amp --lr 0.1
```

To train Highway110:
```commandline
python -u trainer.py --arch=highway110 --amp --wm 0.875
```