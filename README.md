# Weighted Skip Connections are Not Harmful for Deep Nets

Requirement: pytorch (`pip install torch`)

To train the ResNet110 baseline:
```commandline
python -u trainer.py --arch=resnet110 --amp --lr 0.1
```

To train Highway110:
```commandline
python -u trainer.py --arch=highway110 --amp --wm 0.875
```