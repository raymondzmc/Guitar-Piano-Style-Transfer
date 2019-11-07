# Pre-processing
sampling_rate = 44100
n_fft = 2048
hop_length = 512
n_mels = 128


# Resnet configurations
input_nc, output_nc = 1, 1
n_blocks = 9
init_lr = 0.0002
lambda_cyc = 10.0
lambda_idt = 0.5