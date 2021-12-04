# Efficient-conformer


Implementetion of efficient conformer for ASR model from <a href="https://arxiv.org/pdf/2104.06865.pdf">"Efficient conformer-based speech recognition with
linear attention"</a> paper (by Shengqiang Li, Menglong Xu, Xiao-Lei Zhang? CIAIC, School of Marine Science and Technology, Northwestern Polytechnical University, China)

Model files contains in "model" folder in the [efficient_conformer.py](model/efficient_conformer.py) script.

This model was build for classification and ASR tasks. So for launch to train classifier you could run
[train_classifier.py](train_classifier.py) like

```
python3 train_classifier.py
```

To change setting of model you can visit [config.py](config.py) script and set up your
<b>*.csv</b> datasets files.


