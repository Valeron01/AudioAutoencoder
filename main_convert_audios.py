import glob
import os.path

paths = glob.glob("/mnt/LxData/AudioDataset/*.*")

for i in paths:
    target_path = i.replace(os.path.basename(os.path.dirname(i)), "AudioDatasetWAV").replace(i[i.index("."):], ".wav")

    command = f"ffmpeg -i {i} {target_path}"
    print(command)
    os.system(command)
