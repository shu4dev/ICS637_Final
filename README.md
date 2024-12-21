# ICS637_Final

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is the repo for final project of ICS637

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
    ├── AE.py             <- The architecture of the supervised autoencoder model
    │
    ├── config.py               <- Some configuration and imports of libraries.
    │
    ├── dataset.py              <- The PyTorch datamodule used to organize and load the data into the model.
    │
    ├── ingerence.py             <- Code to generate predictions.
    │
    ├── train.py                 <- Code to train the model.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
```

--------

