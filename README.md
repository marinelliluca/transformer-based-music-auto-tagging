# Transformer-based music auto-tagging

Heavily modified from https://github.com/minzwon/sota-music-tagging-models/ 

Here I tried to improve the self-attention architecture for music auto-tagging (https://arxiv.org/abs/1906.04972) by stacking a recurrent layer between the frontend (feature extraction) and the backend (transformer encoder + classifier). It did not improve the performance.

Nonetheless, thanks to various design choices, **the performance of the model was improved from an AUROC of 88.14 to 89.22 on the Million Song Dataset (PRAUC 30.47 --> 32.94), thereby improving the baseline by a considerable margin**.


The best model is located in 'models/best_model_no_recurrent.pth'.
