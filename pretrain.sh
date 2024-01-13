#!/bin/bash
python pretraining.py 100 2 sample_from_all_clusters >> epoch_80_layers_2_all_cls.txt
python pretraining.py 100 4 sample_from_all_clusters >> epoch_80_layers_4_all_cls.txt
python pretraining.py 100 8 sample_from_all_clusters >> epoch_80_layers_8_all_cls.txt
python pretraining.py 100 16 sample_from_all_clusters >> epoch_80_layers_16_all_cls.txt
