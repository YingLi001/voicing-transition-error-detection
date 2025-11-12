# voicing-transition-error-detection

## Description

A research repository for detecting voicing-transition errors in speech signals. The project provides preprocessing pipelines, model training and evaluation code, and inference utilities to identify and classify voicing transition mistakes (onset/offset, voicing flips) from waveform or feature inputs.

## Features

- Data loaders for annotated voicing-transition datasets
- Signal preprocessing: framing, windowing, feature extraction
- Bi-LSTM model for sequence classification
- Training scripts with configurable hyperparameters
- Evaluation scripts producing precision, recall, F1 and confusion matrices
- Inference script for per-file predictions and aggregated reports

## Output & Metrics

- Evaluation produces:
  - Precision / Recall / F1 per class
  - Overall accuracy
  - Time-aligned prediction traces (JSON)
  - Confusion matrix images (PDF)

## Reproducibility

- Set `seed` in to reproduce runs.
- Use deterministic cudnn flags where applicable (see training script).

## License

MIT License — see LICENSE file.

## Contact

Report bugs or request features via the repository's GitHub Issues.