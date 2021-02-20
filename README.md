# Face Anti-spoofing
This is a project for face anti-spoofing, it was done as the final project of **Deep Learning for Computer Vision (Fall, 2020)**. Check out [our poster](poster.pdf) for futher details.

## Run the code

```bash
# Download models file from dropbox
bash download_models.sh

# Predict anomaly score on OULU or SiW test set
bash predict_anomaly.sh [path/to/oulu_test_dir | path/to/siw_test_dir] path/to/output.csv

# Predict class label on OULU or SiW test set
bash predict_class.sh [path/to/oulu_test_dir | path/to/siw_test_dir] path/to/output.csv
```