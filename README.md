# MIAS_BreastCancer

The code uses pre-trained VGG as feature extractor and also add attention map weights on feature responses.

The total number of data points with annotated B and M is only 123. I split the data into train/test in a ratio of 0.9/0.1 (stratified on severity).

The number of data doesn't support a VGG finetuning. So, I froze the VGG layers and only trained the top fully connected layers.

The final predict results are bad. I tried to add a Gaussian mask around the abnormal center to improve the results, but it doesn't help.

To see the result, run eval.py.
