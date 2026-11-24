DATA ?= data/games2p2k.csv
NAME := $(basename $(notdir $(DATA)))

.PHONY: test synthetic prosocial models classic figures all

test:
	python -m pytest -q tests

# Pipeline check on synthetic data from a known cognitive hierarchy model
synthetic:
	python scripts/run_models.py
	python scripts/make_figures.py results/synthetic

# Synthetic data with prosocial players the classic models can't describe
prosocial:
	python scripts/run_models.py --truth prosocial
	python scripts/make_figures.py results/synthetic_prosocial

models:
	python scripts/run_models.py --data $(DATA)

classic:
	python scripts/run_models.py --data $(DATA) --classic-only

figures:
	python scripts/make_figures.py results/$(NAME)

all: test models figures
