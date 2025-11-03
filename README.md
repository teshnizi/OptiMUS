# **OptiMUS**: Optimization Modeling Using mip Solvers and large language models




#### Live demo: https://optimus-solver.com/


This repository contains the official implementation for :



[OptiMUS: Optimization Modeling Using mip Solvers and large language models](https://arxiv.org/pdf/2310.06116).

[OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models](https://arxiv.org/pdf/2402.10172).

[OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale](https://arxiv.org/abs/2407.19633)

<details>
<summary>High-level overview of steps for V0.3</summary>

| Step                                           | Template Used                                      | Inputs (from State)                                                                                               | Additional Inputs                                | State Entries Used                                                                          | State Entries Updated                                                                                                |
| ---                                            | ---                                                | ---                                                                                                               | ---                                              | ---                                                                                         | ---                                                                                                                  |
| 1. Extract Objective Description               | `prompt_objective`                                 | `state["description"]`, `state["parameters"]`                                                                     | `model`, `check`, `logger`, `rag_mode`, `labels` | `state["description"]`, `state["parameters"]`                                               | `state["objective"]["description"]`, `state["objective"]["formulation"] = None`, `state["objective"]["code"] = None` |
| 2. Extract Constraints                         | `prompt_constraints`                               | `state["description"]`, `state["parameters"]`                                                                     | Same as above                                    | `state["description"]`, `state["parameters"]`                                               | `state["constraints"]` (List of constraints with:  `"description"`, `"formulation" = None`, `"code" = None`)         |
| 3. Formulate Constraints Mathematically        | `prompt_constraints_model`                         | `state["description"]`, `state["parameters"]`, `state["constraints"]`, `state.get("variables", {})`               | Same as above                                    | `state["description"]`, `state["parameters"]`, `state["constraints"]`, `state["variables"]` | `state["constraints"][i]["formulation"]`, `state["variables"]` (updated with new variables)                          |
| 4. Formulate Objective Mathematically          | `prompt_objective_model`                           | `state["description"]`, `state["parameters"]`, `state["variables"]`, `state["objective"]`                         | Same as above                                    | `state["objective"]`, `state["variables"]`                                                  | `state["objective"]["formulation"]`                                                                                  |
| 5. Generate Code for Constraints and Objective | `prompt_constraints_code`, `prompt_objective_code` | `state["description"]`, `state["parameters"]`, `state["variables"]`, `state["constraints"]`, `state["objective"]` | `directions`, `solver`, `model`, `check`         | `state["constraints"]`, `state["objective"]`                                                | `state["constraints"][i]["code"]`, `state["objective"]["code"]`                                                      |
| 6. Execute and Debug Code                      | `debug_template`                                   | `state["description"]`                                                                                            | `dir`, `model`, `logger`, `max_tries`            | N/A                                                                                         | N/A (Code execution and potential updates to code files during debugging)                                            |
</details>

![optimus_agent](https://github.com/teshnizi/OptiMUS/assets/48642434/d4620f46-8742-4827-bb65-2735d854576f)

## NLP4LP Dataset

You can download the dataset from [https://huggingface.co/datasets/udell-lab/NLP4LP](https://huggingface.co/datasets/udell-lab/NLP4LP). Please note that NLP4LP is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.



#### References

**OptiMUS** has two available implementations

**OptiMUS v1** adopts a sequential work-flow implementation. Suitable for small and medium-sized problems.

```
@article{ahmaditeshnizi2023optimus,
  title={OptiMUS: Optimization Modeling Using mip Solvers and large language models},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Udell, Madeleine},
  journal={arXiv preprint arXiv:2310.06116},
  year={2023}
}
```

**OptiMUS v2** adopts agent-based implementation. Suitable for large and complicated tasks.

```
@article{ahmaditeshnizi2024optimus,
  title={OptiMUS: Scalable Optimization Modeling with (MI) LP Solvers and Large Language Models},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Udell, Madeleine},
  journal={arXiv preprint arXiv:2402.10172},
  year={2024}
}
```

**OptiMUS v3** adds RAG and large-scale optimization techniques. 

```
@article{ahmaditeshnizi2024optimus,
  title={OptiMUS-0.3: Using Large Language Models to Model and Solve Optimization Problems at Scale},
  author={AhmadiTeshnizi, Ali and Gao, Wenzhi and Brunborg, Herman and Talaei, Shayan and Udell, Madeleine},
  journal={arXiv preprint arXiv:2407.19633},
  year={2024}
}
```





