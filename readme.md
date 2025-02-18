## VL-models for zero-shot instructed vision-to-language generation (Encoder-decoder architecture)


**Install**

Require python>=3.9

~~~bash
pip install salesforce-lavis
pip install opencv-python
~~~

If `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` is raised:

~~~bash
apt-get update -y
apt-get install -y libgl1-mesa-glx
apt-get install -y libglib2.0-0
~~~

We implemented our code based on the paper: Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers

[https://github.com/hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)

Please see the `examples.ipynb`.