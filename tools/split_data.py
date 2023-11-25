import os
import random
import shutil

root_folder = '/home/lacie/Datasets/KISA/train'

event_folders = [
    'Abandonment', 'Falldown', 'FireDetection', 'Intrusion',
    'Loitering', 'PeopleCounting', 'Queueing', 'Violence'
]

# Define the ratio for splitting (80% train and 20% validation)
train_ratio = 0.8

train_folder = '/home/lacie/Datasets/KISA/ver-2/train'
val_folder = '/home/lacie/Datasets/KISA/ver-2/val'

if not os.path.exists(train_folder):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

for event in event_folders:
    event_folder = os.path.join(root_folder, event)
    videos = [file for file in os.listdir(event_folder) if file.endswith('.mp4')]

    random.shuffle(videos)
    num_train = int(len(videos) * train_ratio)

    train_videos = videos[:num_train]
    val_videos = videos[num_train:]

    for video in train_videos:
        xml_file = video.replace('.mp4', '.xml')
        shutil.copyfile(os.path.join(event_folder, video), os.path.join(train_folder, event, video))
        shutil.copyfile(os.path.join(event_folder, xml_file), os.path.join(train_folder, event, xml_file))

    for video in val_videos:
        xml_file = video.replace('.mp4', '.xml')
        shutil.copyfile(os.path.join(event_folder, video), os.path.join(val_folder, event, video))
        shutil.copyfile(os.path.join(event_folder, xml_file), os.path.join(val_folder, event, xml_file))


