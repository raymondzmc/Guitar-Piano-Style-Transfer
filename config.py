# Path
piano = 'piano'
guitar = 'guitar'
ckpt = 'checkpoints'
data_path = 'data/processed'

# Pre-processing
sampling_rate = 16000
n_fft = 1024
hop_length = 256
n_mels = 128
db_min = -15 # Minimum threshold when converting spectrogram to decibel
db_max = 65 # Maximum threshold when converting spectrogram to decibel

# Dataloader
batch_size = 1
n_workers = 0

# Resnet configurations
input_nc, output_nc = 1, 1
n_blocks = 9
lambda_cyc = 10.0
lambda_idt = 0.0

# Training
init_lr = 0.0002
epoch = 4800