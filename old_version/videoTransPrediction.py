from video_transformers import VideoModel

model = VideoModel.from_pretrained("/home/lacie/Github/AbnormalBehaviorRecognition/runs/exp2/checkpoint")

model.predict(video_or_folder_path="/home/lacie/Datasets/KISA/ver-3/slowfast/val/Violence/C104104_006_11.mp4")

