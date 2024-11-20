# Personal Setup Instructions; 

1. Install venv using `uv` if doesn't exist already.
```bash 
uv venv --python 3.10.12
```

2. Install dependencies using: 
```bash 
uv pip install -e .
```

### Important Files

You can find an example of running the evaluation in `eval.py`. 
All the benchmarks are defined in `unibench/benchmarks_zoo/benchmarks.py`. The `