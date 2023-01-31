# nlp-review-summarization

Get the data:
The data can be obtained at https://www.dropbox.com/sh/yzsjwwb698wt546/AAD-ZWDndR2tfECD8PFdr1daa?dl=0.
Download `data.csv` and `data-sentiment-normalised.csv` to this folder.
Download `checkpoint-42000` into the `model` folder.

Usage:
```
python main.py --help
```

For example:
```
python main.py --random
```
or 
```
python main.py --search "Intouchables"
```

If you want to use `--common-embedding` and did not download the checkpoint provided in the dropox, you have to train the classification head first by running
```
python train_classifier.py
```