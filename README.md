Implementation of Deep LoRA in Flax/JAX.

## Setup
1. Create and activate virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
2. Install `dlt` as a local package
```
pip install -e .
```

## Experiments
All experiments from the paper can be found in the `scripts` folder.

- To add a new model, extend the enum in `configs.py` and add LoRA parameter paths to `model_utils.py`. 
- To add a new dataset, extend the enums in `configs.py` and create a new load function in `data.py` (may also need to add a metric to `metrics.py`).

## Known Issues
- Setting `lora_compress=True` can cause OOM errors, this will be fixed soon.
