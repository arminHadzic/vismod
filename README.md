# VisMod
Vismod is a neural network–powered tool designed to automatically moderate visual content, ensuring image data is appropriate for general audiences. Designed for speed and scalability, it helps businesses automate content safety using modern computer vision techniques.


# How To

## Run Docker
Choose whether to run with a CPU or GPU.

### Use CPU-Only:
```bash
	# From project root (vismod/)
	make build-cpu     # build slim version
	make run-cpu       # run on CPU
```
### Use GPU:
```bash
	# From project root (vismod/)
	make build-gpu     # CUDA-optimized build
	make run-gpu       # run with GPU support
```

### Docker Image Cleanup (Optional):
```bash
	# From project root (vismod/)
	make clean         # optional image cleanup
```

## Train the Model:
1. Structure your data in this manner:
your-data-dir/
	├── train/
	│   ├── safe/
	│   └── not/
	├── val/        # optional
	│   ├── safe/
	│   └── not/
	├── test/       # optional
	    ├── safe/
	    └── not/

2. Run a training job:
```bash
	make train
```
### Optional Commands
- Specify a new model
```bash
	# With a different config file
	make train CONFIG=configs/alt_model.yaml
```
- Force CPU
```bash
	make train USE_GPU=0   # Force CPU
```
- Force GPU
```bash
	make train USE_GPU=1   # Force GPU
```
- Specify a data directory
```bash
	# With a different config file
	make train DATA_DIR=your-data-dir
```

## Run Inference:
Run on 1 image or an entire directory:
```bash
	make infer CHECKPOINT=my.ckpt INPUT=my_data BATCH=1000 WORKERS=16
```
### Optional Commands
- Custom Output Directory
```bash
	make infer OUTPUT=my_dir # Custom output directory
```
- Force CPU
```bash
	make infer USE_GPU=0   # Force CPU
```
- Force GPU
```bash
	make infer USE_GPU=1   # Force GPU
```

## Shell into Container (GPU if Available):
```bash
	make run
```

## Miscellaneous:
To re-lint the code:
```bash
	# From project root (vismod/)
	make format        # format with yapf (2-space config)
```