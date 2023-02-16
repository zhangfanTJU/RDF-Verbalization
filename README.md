# RDF-Verbalization

This repository contains the corpus and code for **RDF-Verbalization**.

[**Enhancing RDF Verbalization with Descriptive and Relational Knowledge**](https://github.com/zhangfanTJU/RDF-Verbalization)

Fan Zhang, Meishan Zhang, Shuang Liu and Nan Duan

## Reqirements
```
pip install -r requirements.txt
```

## Downloading Checkpoints
We use the following models: 
- [BART-Base](https://huggingface.co/facebook/bart-base)
- [T5-Small](https://huggingface.co/google/t5-v1_1-small)
- [T5-Base](https://huggingface.co/google/t5-v1_1-base)


## Downloading Datasets
We use the following datasets: 
- [WebNLG](https://webnlg-challenge.loria.fr/challenge_2017/)
- [SemEval-2010 Task 8](https://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11)

## Preprocessing
For preprocessing the WebNLG dataset, run:
```
python3 data/generate_input_webnlg.py <dataset_folder>
```

## Finetuning
For finetuning the models using the WebNLG dataset, run:
```
# Baseline
bash scripts/train-t5.sh

# +D
bash scripts/train-t5-def.sh

# +R
bash scripts/train-t5-kg.sh

# +Both
bash scripts/train-t5-kd.sh
```

## Decoding
For decoding using beam search, run:
```
# Baseline
bash scripts/test-t5.sh

# +D
bash scripts/test-t5-def.sh

# +R
bash scripts/test-t5-kg.sh

# +Both
bash scripts/test-t5-kd.sh
```
We use the `--eval_beams` flag to specify the beam size (default is `3`).

## Citation

```bibtex
@article{zhang2022rdf,
  title={Enhancing RDF Verbalization with Descriptive and Relational Knowledge}, 
  author={Fan Zhang, Meishan Zhang, Shuang Liu and Nan Duan}
}
```
