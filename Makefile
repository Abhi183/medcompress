.PHONY: install test lint figures clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --tb=short

lint:
	python -m py_compile models/baseline.py
	python -m py_compile compression/qat.py
	python -m py_compile compression/distillation.py
	python -m py_compile compression/pruning.py
	python -m py_compile compression/sparse_attention.py
	python -m py_compile data/isic_loader.py
	python -m py_compile data/brats_loader.py
	python -m py_compile data/chexpert_loader.py
	python -m py_compile data/kvasir_loader.py

figures:
	python figures/generate_figures.py
	python figures/generate_eda.py

# Train baselines
train-isic:
	python scripts/train.py --config configs/isic_baseline.yaml

train-brats:
	python scripts/train.py --config configs/brats_baseline.yaml

train-chexpert:
	python scripts/train.py --config configs/chexpert_baseline.yaml

train-kvasir:
	python scripts/train.py --config configs/kvasir_baseline.yaml

# Compress
compress-isic-qat:
	python scripts/compress.py --config configs/isic_qat.yaml

compress-isic-kd:
	python scripts/compress.py --config configs/isic_kd.yaml

# Evaluate
evaluate:
	python scripts/evaluate.py --config configs/isic_qat.yaml

benchmark:
	python scripts/benchmark_runtime.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache outputs/ dist/ build/ *.egg-info
