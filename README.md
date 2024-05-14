# **OptiMUS**: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models

#### Live demo: https://optimus-solver.vercel.app/

This repository contains the official implementations for [OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models](https://arxiv.org/pdf/2402.10172). Check out [this](https://github.com/teshnizi/OptiMUS/tree/optimus_v1) branch for an implementation of the older version.

![AgentTeam](https://github.com/teshnizi/OptiMUS/assets/48642434/ae11ff0d-2d1e-4832-9dcc-533af4c5cde0)

## NLP4LP Dataset

You can download the dataset from https://nlp4lp.vercel.app/. Please note that NLP4LP is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes (The updated version will be added soon).

## Running the code

### First install the requirement:

```bash
pip install -r requirements.txt
```

### Install gurobi and your license (for gurobi installation, please refer to the official website):

```bash
grbgetkey YOUR_LICENSE
```

### Add your API keys to config.py:

```python
{
    "openai_api_key": "OPENAI_API_KEY",
    "openai_org_id": "OPENAI_ORG_ID",
    "together_api_key": "TOGETHER_API_KEY",
    "mistral_api_key": "MISTRAL_API_KEY",
}
```

### Download and add the data to the repo in the following structure:

```
OptiMUS
│   README.md
│   run.py
│   requirements.txt
│   config.json
│   LICENSE
│   agents/
│   data/
│       nlp4lp/
│       complexor/
│       nl4opt/
```

NLP4LP is available [here](https://nlp4lp.vercel.app/). ComplexOR and NLP4LP datasets are available here (in the supplementary material): https://openreview.net/forum?id=HobyL1B9CZ

### Run the script:

```bash
python run.py
```

### You can modify the arguments to run the script on different datasets, models, and problems. For example:

```bash
 python run.py --dataset nlp4lp --problem 1
```

or

```bash
 python run.py --dataset complexor --problem Knapsack
```

## Reference

#### Have questions? Want to implement this idea? Feel free to reach out via email (teshnizi /at/ stanford /dot/ edu)

#### Reference

```
@article{ahmaditeshnizi2024optimus,
  title={OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Udell, Madeleine},
  journal={arXiv preprint arXiv:2402.10172},
  year={2024}
}
```
