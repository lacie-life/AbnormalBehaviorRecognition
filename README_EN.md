# Development_of_abnormal_Behavior_Recognition

hello.AIRLAB is Roh Hyun -chul.

I am 2021.01.25.~ 2021.02.05.Until the Ministry of Science and Technology and the Korea Information Society Agency, we participated in <Identification of Subway Station CCTV Video> and achieved test set ** 97.5%** performance. <br>
Reference: [link](http://aifactory.space/task/detail.do?taskid=t001632)

## 1. Development of abnormal behavioral awareness algorithm
### 1.1.Background
 This hackathon is to use the images of CCTVs installed in the subway station to improve the completeness of the project to detect abnormal behavior and to track objects, and to promote the performance of the project.
 <br>
 In most subway history, CCTVs are being used to prevent safety accidents and social crimes and to support the weak.Since CCTV, currently in history, is passively monitored by station attendants, so it is limited to usual use.Therefore, this conference was planned for the purpose of preventing safety accidents through intelligent CCTVs in the subway history, detecting social crimes early and providing improved services to consumers. <br>
 <br>
In this competition, the performance results of the AI model are evaluated by classifying five ideal behaviors in the subway into five (organic, escalator evangelism, fainting, environmental evangelism, theft).

<P align = "Center"> <IMG SRC = "https://user-images.githubusercontent.com/53032349/107145082-b64b8680-6982-11eb-8ee3-a71e026f92c3.PNG" widt "width = "90%" height = "90%"Title =" 70px "alt =" MemoryBlock "> </P>

## 2. Model development process
 The model development process was written in order of experiments. <br>
### 2.1.Baseline Model
 First, the baseLine code provided by the organizer was composed of 3D-resnet (backbone), and to measure performance, both 3D-Resnet50, 3D-Resnet101 experimented with both 3D-resnet50, and 3D-resnet101 with ** 65%**The performance was better.It's a light task, so it's a lighter model than a heavy model.Hyperparameter, such as LR and Batch, was applied based on loss and ACC when experimenting, and fixed to Batch: 32, LR: 0.001. <br>
<br>
 Since then, we have conducted various experiments based on basic performance.The first was to change the model and measure it.In the Baseline code, we changed the backbone to R (2+1) D and experimented.This was changed to senior advice and experimented with R (2+1) in a simple task that it might be good.The result was ** 66 ~ 68%**, which was better than basic baseline code. <br>
<br>
### 2.2.3D MODEL
 Next, I tried to change the backbone to Resnext. <br>
(I thought the data loader part was an error, I took print, and searched the strange error window several times.
In addition, in the model FC part, the output part was changed directly from the model code, so there were many errors (let's touch it with a code that was loaded without touching it directly) <br>
 Therefore, instead of the organizers' Baseline code, it was replaced with [Mars] (https://github.com/craston/mars) code. <br>
 And because I had the first time this competition, I knew that I shouldn't use the prerain model, but I used Kinetics Pretain Model because it was irrelevant.Resnext50, 101 experimented with both, and Resnext50 achieved ** 85%**.The reason why 50 is better than 101 seems to be the same as I mentioned earlier. <br>
<br>
 I used Pretain Model in Mars Resnext50, and the previous experiment was experimented with only the last layer and the last FC.However, when I read the Transfer Learning paper in the past, there was a record that had a better result of finishing the whole, so this experiment was the same as the previous experiment, but I conducted a full fine tuning.The results were better at expected ** 87.5%** performance. <br>
<br>
### 2.3.2D MODEL
 BASELINE code and mars code are 3D-model.But 3D-MODEL is heavier than 2D-MODEL.In addition, it is a simple task, so even if you use 2D, the performance will be good.The 2D-MODEL is modified because it is different from the 3D-MODEL data loader, and only one of the video frames is made and classified.The model used Resnet50 and used Imagenet Pretrain. <br>
The result was a maximum ** 91.3%**, which was much better than 3D-MODEL.Before this, I used only randomhorizontalflip and randomrotation because Tiny Imagenet Challenge had a better transform than excessive transform.Later randomrotation was removed because it did not produce performance.The reason is that there are three labels in the case of a person falling in the data set, and it seemed to make the randomrotation fall.

<p align = "center"> <img src = "https://user-images.githubusercontent.com/53032349/107150249-43e99f00-69a0-11eb-90a8-0b0b21645ce0.PNG" width = "80%" height = "80%"Title =" 70px "alt =" MemoryBlock "> </P>

<br>
 I analyzed Train and Test Dataset and found that the last frame (approximately 30%) was an image that was not related to the last frame (30%), except for the last frame (30%), I used only the remaining 70%, but the performance was the same or rather lower..Because of this, I thought the model also learned the camera's composition. <br>
<br>
 It is the same setting as the experiment, but it was used only the rest (90%, 80%), except for the last frame (10%, 20%), but the results were the same as before.Since then, I have done detailed experiments, such as changing SGD to ADAM, but the performance has been the same or dropped. <br>
<br>
 In 3D-MODEL, the image size was fixed to 112, which was fixed in 2D-Model, but it was tested as it was increased to 224 and 448, and ** 93%and 95.8%** were achieved.I also experimented with another network in Resnet50, but the performance was similar or bad. <br>
<br>
 Finally, Hyperparameter, such as Batch, LR, and Image size, was properly adjusted to achieve the highest performance ** 97.5%**.
 <br>
 <P align = "Center"> <IMG SRC = "https://user-images.githubusercontent.com/53032349/107150378-dbe78880-69a0-11eb-93bc-3d98a22ecad3.PNG" widt "width = "70%" height = "70%"Title =" 70px "alt =" MemoryBlock "> </P>
 
 ## reproduction of performance
 The following process can reproduce model performance.Since 2D-Model uses torchvision.models, there is no need for prerained model, but 3D-model is required, so be careful about performance reproduction. <br>
```Shell
  python train2d.py --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 5 --batch_size 32 --log 1 --sample_duration 64 --model resnet --model_depth 50 --ft_begin_index 0  --result_path "results/" --n_workers 8 --n_epochs 100 --learning_rate 0.01
  
  or
  
  python train3d.py --modality RGB --split 1 --only_RGB --n_classes 400 --n_finetune_classes 5 --batch_size 32 --log 1 --sample_duration 64 --model resnext --model_depth 101 --ft_begin_index 0  --result_path "results/" --n_workers 8 --n_epochs 100 --learning_rate 0.01
 ```
