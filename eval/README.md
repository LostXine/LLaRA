# Evaluation on VIMABench

This page will guide you to evaluate your trained LLaVA model on [VIMABench](https://github.com/vimalabs/VIMABench).

## Quick Start

Evaluate models trained on `inBC / D-inBC`:
```
python3 eval-llara.py [evaluation name] --model-path [MODEL_PATH]
```

Evaluate models trained on `RT-2` style datasets:
```
python3 eval-rt2.py [evaluation name] --model-path [MODEL_PATH]
```

## Detailed Usage

Usage: 
```
python3 eval-llara.py [-h] [--model-path MODEL_PATH] [--output-path OUTPUT_PATH] [--prompt-mode PROMPT_MODE] [--seed SEED] [--num-env NUM_ENV] [--max-length MAX_LENGTH] [--partition PARTITION] [--detector DETECTOR] [--detector-thre DETECTOR_THRE] filename
```

- `filename`: Name of the output file.
- `OUTPUT_PATH`: Path to the output directory (default: `../results/`).
- `MODEL_PATH`: Path to LLaVA checkpoint.
- `PROMPT_MODE`: Set of operational flags:
- - `h`: Enable action history.
- - `s`: Query VLM for each observation and perform a single action step no matter how many steps the VLM generates.
- - `d`: Enable object detection using VLM.
- - `e`: Enable object detection using MaskRCNN.
- - `o`: Enable oracle object detection.
- `SEED`: Random seed for reproducibility.
- `NUM_ENV`: Number of episodes per task.
- `MAX_LENGTH`: Maximum steps per episode; episodes exceeding this limit are marked as failed.
- `PARTITION`: Specific partition of VIMABench to test; tests all partitions (L1 - L4) if unspecified.
- `DETECTOR`: Path to the MaskRCNN checkpoint, used only if `e` is enabled in `PROMPT_MODE`.
- `DETECTOR_THRE`: Minimum score for an object detection proposal to be considered valid (default: 0.6).

For models trained on datasets without object detection features, use `PROMPT_MODE` as `hs`.

For models starting with `D-`, set `PROMPT_MODE` as `hsd`, `hse`, or `hso` to enable different object detectors.

Note: Both `eval-llara.py` and `eval-rt2.py` scripts accept the same arguments.
