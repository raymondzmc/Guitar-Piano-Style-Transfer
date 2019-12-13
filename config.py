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

x_min = -6.5
x_max = 0.9
y_min = -5.7
y_max = 1.1

db_max = 65

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
epoch = 2000