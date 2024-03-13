SHELL := /bin/bash

isort:
	isort .

black:
	brunette .

clean: isort black

lint:
	isort --check-only . --diff
	flake8 --show-source .
	brunette . --check -t py38

build_main:
	DOCKER_BUILDKIT=1 docker build -t no_sfm_gaussian_splattings .

build:
	$(MAKE) build_main

run:
	docker run -d -it --init --runtime=nvidia \
	--gpus=all \
	--ipc=host \
	--volume=$(CURDIR):/app \
	--volume=/home:/hhome \
	--publish="5555:5555" \
	--publish="5565:5565" \
	no_sfm_gaussian_splattings bash