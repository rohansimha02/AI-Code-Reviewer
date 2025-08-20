.PHONY: setup data bugsinpy quixbugs split stats train eval serve demo test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data: bugsinpy quixbugs split stats

bugsinpy:
	python scripts/01_extract_bugsinpy.py --raw_dir data/raw/bugsinpy --out data/interim/bugsinpy_pairs.parquet --max_chars 4000 --context_lines 15

quixbugs:
	python scripts/02_extract_quixbugs.py --raw_dir data/raw/quixbugs --out data/interim/quixbugs_pairs.parquet --max_chars 4000

split: 
	python scripts/03_merge_and_split.py

synthetic: 
	python scripts/05_generate_synthetic_bugs.py --input data/processed/train.parquet --output data/interim/synthetic_bugs.parquet --num_variants 3

merge_synthetic:
	python scripts/06_merge_synthetic.py --train data/processed/train.parquet --synthetic data/interim/synthetic_bugs.parquet --output data/processed/train_with_synthetic.parquet

stats: 
	python scripts/04_show_stats.py

train:
	python training/train.py --config training/config.yaml

eval:
	python training/evaluate.py --config training/config.yaml

serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

demo:
	streamlit run demo/streamlit_app.py

test:
	pytest -q

clean:
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf serving/model_store/*
	rm -rf .venv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
