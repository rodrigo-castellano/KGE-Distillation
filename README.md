# Distilling KGE Black Boxes into Interpretable NeSy Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24465162.svg)](https://doi.org/10.5281/zenodo.24465162)

Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning (NeSy 2025), PMLR vol. 284, pp. 736–749, 2025  
[Paper page](https://rodrigocastellano.com/papers/distilling-kge/) · [PDF](https://rodrigocastellano.com/papers/distilling-kge/distilling-kge-nesy2025.pdf) · [Official version](https://proceedings.mlr.press/v284/ontiveros25a.html)


This repository contains the code for the paper: **Distilling KGE black boxes into interpretable NeSy models** by Rodrigo Castellano Ontiveros, Francesco Giannini, and Michelangelo Diligenti, accepted to the 19th International Workshop on Neural-Symbolic Learning and Reasoning (NeSy 2025).

## Training

To train the models presented in the paper, navigate to the `experiments` folder and run the `runner.py` script.
**Example:**
```bash
python experiments/runner.py --d countries_s3 --m sbr --resnet False --distill True
```
## Data
The datasets can be found in the folder experiments/data.

## Theory
Theoretical foundations can be found in the manuscript: **Distilling KGE black boxes into interpretable NeSy models**, by Rodrigo Castellano Ontiveros, Francesco Giannini, and Michelangelo Diligenti.

## Authors
Rodrigo Castellano Ontiveros and Michelangelo Diligenti.

## Licence
Copyright 2020

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

## Citation

If you use this code, please cite the paper below. The code itself is archived on Zenodo: [doi:10.5281/zenodo.24465162](https://doi.org/10.5281/zenodo.24465162).

```bibtex
@inproceedings{castellanoontiveros2025distilling,
  title     = {Distilling {KGE} black boxes into interpretable {NeSy} models},
  author    = {Castellano Ontiveros, Rodrigo and Giannini, Francesco and Diligenti, Michelangelo},
  booktitle = {Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {284},
  pages     = {736--749},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v284/ontiveros25a.html}
}
```
