# Project-wide variables
IMAGE_NAME=vismod
CPU_TAG=$(IMAGE_NAME):cpu
GPU_TAG=$(IMAGE_NAME):gpu

# These are all overridable
DATA_DIR ?= data
CONFIG ?= configs/mod_classifier.yaml
CHECKPOINT ?= checkpoints/model.ckpt
INPUT ?= images_to_check
WORKERS ?= 4
SHM_SIZE ?= 1g

# Default output directory with timestamp
NOW := $(shell date +"%Y-%m-%d_%H-%M-%S")
DEFAULT_OUTPUT := inference_results_$(NOW)
OUTPUT ?= $(DEFAULT_OUTPUT)

# Detect whether GPU is available
GPU_AVAILABLE := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
USE_GPU ?= $(GPU_AVAILABLE)

# Select image and runtime
TAG := $(if $(filter 1,$(USE_GPU)),$(GPU_TAG),$(CPU_TAG))
RUNTIME := $(if $(filter 1,$(USE_GPU)),--gpus all,)

# Logging verbosity
VERBOSE ?= 0
LOG_LEVEL := $(if $(filter 1,$(VERBOSE)),INFO,WARNING)

.ONESHELL:
SHELL := /bin/bash

# ---------- Docker Build ----------

build-cpu:
	docker build -f docker/Dockerfile -t $(CPU_TAG) .

build-gpu:
	docker build -f docker/Dockerfile.gpu -t $(GPU_TAG) .

# ---------- Interactive Shell ----------

run:
	docker run $(RUNTIME) -it --rm -v $$PWD:/app $(TAG)



# ---------- Training ----------

train:
	@if [ -z "$(DATA_DIR)" ]; then \
	  echo "ERROR: DATA_DIR is not set"; exit 1; \
	fi; \
	MOUNT_SRC="$(DATA_DIR)"; \
	MOUNT_DEST="/app/data"; \
	[ "$$(echo $(DATA_DIR) | cut -c1)" != "/" ] && MOUNT_SRC="$$PWD/$(DATA_DIR)"; \
	echo "Mounting $$MOUNT_SRC to $$MOUNT_DEST"; \
	docker run $(RUNTIME) --rm \
	  --shm-size=$(SHM_SIZE) \
	  -v "$$MOUNT_SRC:$$MOUNT_DEST" \
	  -v "$$PWD:/app" \
	  $(TAG) \
	  python src/train.py \
	    fit \
	    --config /app/$(CONFIG) \
	    --data.data_dir=$$MOUNT_DEST \
	    --log_level="$(LOG_LEVEL)"

# ---------- Evaluate ----------

REPORT ?= 0 # 1 = generate PR curve + README table
EVAL_BATCH_SIZE ?= 10

eval:
	@set -euo pipefail
	if [[ -z "$(CHECKPOINT)" || -z "$(DATA_DIR)" ]]; then
	  echo "ERROR: CHECKPOINT and DATA_DIR must be specified"; exit 1;
	fi

	# Normalize to absolute paths on host
	CKPT="$(CHECKPOINT)"; DATA="$(DATA_DIR)"
	case "$$CKPT" in /*) ;; *) CKPT="$$PWD/$$CKPT" ;; esac
	case "$$DATA" in /*) ;; *) DATA="$$PWD/$$DATA" ;; esac

	# Container-local paths
	LOG_FILE_C="/app/eval_log.log"
	REPO_DIR_C="/app"

	# Optional report flags
	EXTRA=""
	if [[ "$(REPORT)" == "1" ]]; then
	  EXTRA="--make_report --repo_dir=$$REPO_DIR_C"
	fi

	# Mount dirs (checkpoint dir may be outside repo)
	DIR_CKPT="$$(dirname "$$CKPT")"

	docker run $(RUNTIME) --rm --shm-size=$(SHM_SIZE) \
	  -v "$$PWD:/app" \
	  -v "$$DIR_CKPT:$$DIR_CKPT" \
	  -v "$$DATA:$$DATA" \
	  $(TAG) \
	  python src/eval.py \
	    --checkpoint="$$CKPT" \
	    --data_dir="$$DATA" \
	    --batch_size=$(EVAL_BATCH_SIZE) \
	    --num_workers=$(WORKERS) \
	    --log_file="$$LOG_FILE_C" \
	    --log_level="$(LOG_LEVEL)" \
	    $$EXTRA



# ---------- Inference ----------

SAMPLES_PER_FILE ?= 1000

infer:
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(INPUT)" ]; then \
	  echo "ERROR: CHECKPOINT and INPUT must be specified"; exit 1; \
	fi; \
	CHECKPOINT_PATH="$(CHECKPOINT)"; \
	INPUT_PATH="$(INPUT)"; \
	OUTPUT_PATH="$(OUTPUT)"; \
	LOG_FILE="$$(OUTPUT)/infer_log.log"; \
	[ "$$(echo $(CHECKPOINT) | cut -c1)" != "/" ] && CHECKPOINT_PATH="$$PWD/$(CHECKPOINT)"; \
	[ "$$(echo $(INPUT) | cut -c1)" != "/" ] && INPUT_PATH="$$PWD/$(INPUT)"; \
	[ -z "$(OUTPUT)" ] && OUTPUT_PATH="inference_results_$$(date +%F_%H-%M-%S)"; \
	[ "$$(echo $$OUTPUT_PATH | cut -c1)" != "/" ] && OUTPUT_PATH="$$PWD/$$OUTPUT_PATH"; \
	echo "Using checkpoint: $$CHECKPOINT_PATH"; \
	echo "Reading images from: $$INPUT_PATH"; \
	echo "Saving results to: $$OUTPUT_PATH"; \
	MOUNT_CHECKPOINT_DIR=$$(dirname "$$CHECKPOINT_PATH"); \
	MOUNT_INPUT_DIR=$$(dirname "$$INPUT_PATH"); \
	docker run $(RUNTIME) --rm \
	  --shm-size=$(SHM_SIZE) \
	  -v "$$PWD:/app" \
	  -v "$$MOUNT_CHECKPOINT_DIR:$$MOUNT_CHECKPOINT_DIR" \
	  -v "$$MOUNT_INPUT_DIR:$$MOUNT_INPUT_DIR" \
	  -v "$$OUTPUT_PATH:$$OUTPUT_PATH" \
	  $(TAG) \
	  python src/infer.py \
	    --checkpoint="$$CHECKPOINT_PATH" \
	    --input="$$INPUT_PATH" \
	    --output="$$OUTPUT_PATH" \
	    --samples_per_file=$(or $(SAMPLES_PER_FILE),500) \
	    --num_workers=$(or $(WORKERS),4) \
	    --log_file="$$LOG_FILE" \
	    --log_level="$(LOG_LEVEL)"


# ---------- Notebooks ----------

NB_PORT ?= 8888

notebook:
	docker run $(RUNTIME) --rm --shm-size=$(SHM_SIZE) \
	  -p $(NB_PORT):8888 \
	  -v "$$PWD:/app" \
	  $(TAG) \
	  bash -lc 'jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""'


# ---------- Formatting ----------

format:
	@TAG=$$(docker images -q vismod:cpu); \
	if [ -n "$$TAG" ]; then \
	  IMAGE=vismod:cpu; \
	elif docker images -q vismod:gpu > /dev/null; then \
	  IMAGE=vismod:gpu; \
	else \
	  echo "ERROR: Neither vismod:cpu nor vismod:gpu image found."; exit 1; \
	fi; \
	echo "Using image: $$IMAGE"; \
	docker run --rm -v "$$PWD:/app" $$IMAGE yapf -ir /app/src
		docker run --rm -v "$$PWD:/app" $$IMAGE yapf -ir /app/scripts


# ---------- Make Demo Dataset ----------

build-dataset:
	@if [ -z "$(DATA_DIR)" ]; then \
	  echo "ERROR: DATA_DIR is not set"; exit 1; \
	fi; \
	MOUNT_SRC="$(DATA_DIR)"; \
	MOUNT_DEST="/app/data"; \
	[ "$$(echo $(DATA_DIR) | cut -c1)" != "/" ] && MOUNT_SRC="$$PWD/$(DATA_DIR)"; \
	echo "Mounting $$MOUNT_SRC to $$MOUNT_DEST"; \
	docker run $(RUNTIME) --rm \
	  --shm-size=$(SHM_SIZE) \
	  -v "$$MOUNT_SRC:$$MOUNT_DEST" \
	  -v "$$PWD:/app" \
	  $(TAG) \
	  python scripts/prepare_imagenette.py \
	    --out=$$MOUNT_DEST


# ---------- Cleanup ----------

clean:
	docker rmi $(CPU_TAG) $(GPU_TAG) || true
