# Project-wide variables
IMAGE_NAME=vismod
CPU_TAG=$(IMAGE_NAME):cpu
GPU_TAG=$(IMAGE_NAME):gpu

# Training and inference config (overridable)
CONFIG ?= configs/mod_classifier.yaml
CHECKPOINT ?= checkpoints/model.ckpt
INPUT ?= images_to_check
BATCH ?= 1000
WORKERS ?= 8
DATA_DIR ?= data

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

# ---------- Docker Build ----------

build-cpu:
	docker build -f docker/Dockerfile -t $(CPU_TAG) .

build-gpu:
	docker build -f docker/Dockerfile.gpu -t $(GPU_TAG) .

# ---------- Interactive Shell ----------

run:
	docker run $(RUNTIME) -it --rm -v $$PWD:/app $(TAG)

# ---------- Inference ----------

infer:
	@if [ -z "$(CHECKPOINT)" ] || [ -z "$(INPUT)" ]; then \
	  echo "[!] CHECKPOINT and INPUT must be specified"; exit 1; \
	fi; \
	CHECKPOINT_PATH="$(CHECKPOINT)"; \
	INPUT_PATH="$(INPUT)"; \
	OUTPUT_PATH="$(OUTPUT)"; \
	[ "$$(echo $(CHECKPOINT) | cut -c1)" != "/" ] && CHECKPOINT_PATH="$$PWD/$(CHECKPOINT)"; \
	[ "$$(echo $(INPUT) | cut -c1)" != "/" ] && INPUT_PATH="$$PWD/$(INPUT)"; \
	[ -z "$(OUTPUT)" ] && OUTPUT_PATH="inference_results_$$(date +%F_%H-%M-%S)"; \
	[ "$$(echo $$OUTPUT_PATH | cut -c1)" != "/" ] && OUTPUT_PATH="$$PWD/$$OUTPUT_PATH"; \
	echo "[•] Using checkpoint: $$CHECKPOINT_PATH"; \
	echo "[•] Reading images from: $$INPUT_PATH"; \
	echo "[•] Saving results to: $$OUTPUT_PATH"; \
	MOUNT_CHECKPOINT_DIR=$$(dirname "$$CHECKPOINT_PATH"); \
	MOUNT_INPUT_DIR=$$(dirname "$$INPUT_PATH"); \
	docker run $(RUNTIME) --rm \
	  -v "$$PWD:/app" \
	  -v "$$MOUNT_CHECKPOINT_DIR:$$MOUNT_CHECKPOINT_DIR" \
	  -v "$$MOUNT_INPUT_DIR:$$MOUNT_INPUT_DIR" \
	  -v "$$OUTPUT_PATH:$$OUTPUT_PATH" \
	  $(TAG) \
	  python src/infer.py \
	    --checkpoint="$$CHECKPOINT_PATH" \
	    --input="$$INPUT_PATH" \
	    --output="$$OUTPUT_PATH" \
	    --batch_size=$(or $(BATCH_SIZE),500) \
	    --num_workers=$(or $(WORKERS),4)

# ---------- Training ----------

train:
	@if [ -z "$(DATA_DIR)" ]; then \
	  echo "[!] DATA_DIR is not set"; exit 1; \
	fi; \
	MOUNT_SRC="$(DATA_DIR)"; \
	MOUNT_DEST="/app/data"; \
	[ "$$(echo $(DATA_DIR) | cut -c1)" != "/" ] && MOUNT_SRC="$$PWD/$(DATA_DIR)"; \
	echo "[•] Mounting $$MOUNT_SRC to $$MOUNT_DEST"; \
	docker run $(RUNTIME) --rm \
	  --shm-size=2g \
	  -v "$$MOUNT_SRC:$$MOUNT_DEST" \
	  -v "$$PWD:/app" \
	  $(TAG) \
	  python src/train.py \
	    fit \
	    --config /app/$(CONFIG) \
	    --data.data_dir=$$MOUNT_DEST


# ---------- Formatting ----------

format:
	@TAG=$$(docker images -q vismod:cpu); \
	if [ -n "$$TAG" ]; then \
	  IMAGE=vismod:cpu; \
	elif docker images -q vismod:gpu > /dev/null; then \
	  IMAGE=vismod:gpu; \
	else \
	  echo "[✘] Neither vismod:cpu nor vismod:gpu image found."; exit 1; \
	fi; \
	echo "[•] Using image: $$IMAGE"; \
	docker run --rm -v "$$PWD:/app" $$IMAGE yapf -ir /app/src


# ---------- Cleanup ----------

clean:
	docker rmi $(CPU_TAG) $(GPU_TAG) || true
