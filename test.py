import librosa
import os

path = os.path.join('data', 'youtube_guitar')
fname = '25 Favorite Country Instrumental Songs-Wh5a4kKkAIc.wav'

audio, rate = librosa.core.load(os.path.join(path, fname), sr=None)

start = 3
end = 83 * 60 + 32

new_audio = audio[start * rate:end * rate]
write_path = os.path.join(path, '_' + fname)
librosa.output.write_wav(write_path, new_audio, rate)

print(rate)
