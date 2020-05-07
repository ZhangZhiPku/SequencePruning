# SequencePruning
Source code of paper "Sequence Pruning: as a new way to accelerate NLP models"

# Prepare work
-Download glue data from https://gluebenchmark.com/
  unzip all data into data/
-tokenize all data with src/prepare/PrepareAGNEWS4LSTM.py, src/prepare/PrepareGLUE4LSTM.py, src/prepare/PrepareGLUE4BERT.py
  run all 3 scripts respectively.

# Usage
-LSTM sequence pruning:
  run src/runs/RunLSTM4AGNEWS.py or src/runs/RunLSTM4GLUE.py
  pruner is implemented with src/models/compression.py
  LSTM model is implemented with src/models/lstm.py
-BERT sequence pruning
  run src/runs/RunBERT4GLUE.py
  pruner is implemented with src/models/compression.py
  BERT model is implemented with https://github.com/huggingface/transformers

# Result anaylse
-Padding anaylse:
  run src/anaylse/analyse_agnews_padding.py and src/anaylse/analyse_glue_padding.py
