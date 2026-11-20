DATA ?= data/games2p2k.csv
NAME := $(basename $(notdir $(DATA)))

.PHONY: test synthetic classic figures all

test:
	python -m pytest -q tests

synthetic:
	python scripts/run_classic.py
	python scripts/make_figures.py results/synthetic

classic:
	python scripts/run_classic.py --data $(DATA)

figures:
	python scripts/make_figures.py results/$(NAME)

all: test classic figures
