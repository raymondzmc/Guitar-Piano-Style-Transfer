# CycleGAN for Guitar-Piano Tone Color Transfer

## Dataset
| Dataset         | Piano                  | Guitar                |
| --------------  |:----------------------:|:---------------------:|
| Source          | Maestro                | GuitarSet             |
| Size            | 1044 files (156 hours) |   360 files (3 hours) |
| Sampling Rate   | 44100                  |    44100              |

After downloading the processed dataset [here](https://www.google.com)
```
tar -xvf processed.tar
```

Modify `config.py`, so `data_path` refers to the downloaded directory

## Training
After installing all the dependencies in `requirement.txt`, simply run `python train.py` to train the CycleGAN model according to the configurations in `config.py`
