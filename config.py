# Path
piano = 'piano'
guitar = 'guitar'
ckpt = 'checkpoints'
data_path = 'data/processed'


# Pre-processing
sampling_rate = 44100
n_fft = 2048
hop_length = 512
n_mels = 256
db_min = -15 # Minimum threshold when converting spectrogram to decibel
db_max = 65 # Maximum threshold when converting spectrogram to decibel

# Dataloader
batch_size = 2


# Resnet configurations
input_nc, output_nc = 1, 1
n_blocks = 9
lambda_cyc = 10.0
lambda_idt = 0.5

# Training
init_lr = 0.0002
epoch = 200