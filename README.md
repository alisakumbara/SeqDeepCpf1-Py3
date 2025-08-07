# SeqDeepCpf1-Py3

This repository contains a Python 3 version of the **Seq-deepCpf1** model, originally developed using Keras in Python 2, now implemented in **PyTorch**.

> Only the **Seq-deepCpf1** model has been converted. The **DeepCpf1** model from the original repository is not included.

## Background

The model was originally published in the following paper:

> **Kim, H., Min, S., Song, M. et al.**  
> *Deep learning improves prediction of CRISPR–Cpf1 guide RNA activity.*  
> *Nature Biotechnology*, 36, 239–241 (2018).  
> [https://doi.org/10.1038/nbt.4061](https://doi.org/10.1038/nbt.4061)

Original code: [https://github.com/MyungjaeSong/Paired-Library](https://github.com/MyungjaeSong/Paired-Library)

## Usage

### Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Run the model

```bash
python SeqDeepCpf1.py input_example.txt output_example.txt
```