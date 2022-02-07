# RDF-Verbalization

This repository contains the corpus and code for **RDF-Verbalization**.

[**Enhancing RDF Verbalization with Descriptive and Relational Knowledge**](https://github.com/zhangfanTJU/RDF-Verbalization)

Fan Zhang, Meishan Zhang, Shuang Liu and Nan Duan

## Reqirements
```
pip install -r requirements.txt
```

## Quick Start
```
# T5
## baseline
bash scripts/train-t5.sh

## +D
scripts/train-t5-def.sh

## +R
scripts/train-t5-kg.sh

## +Both
scripts/train-t5-kd.sh
```

## Citation

```bibtex
@article{zhang2022rdf,
  title={Enhancing RDF Verbalization with Descriptive and Relational Knowledge}, 
  author={Fan Zhang, Meishan Zhang, Shuang Liu and Nan Duan}
}
```
