
# Object Detecion 
import cv2
from ultralytics import YOLO
#plots
import matplotlib.pyplot as plt
import seaborn as sns

#basics
import pandas as pd
import numpy as np
import os
import subprocess

from tqdm import tqdm

# Display image and videos
import IPython
from IPython.display import Video, display
# %matplotlib inline

# Video  path for experiment
import gdown

video_file = r"C:\Users\tando\OneDrive\Desktop\New folder\Sample-Recording-14mins - Trim.mp4"
# video_file = 'input_video.mp4'
# gdown.download(video_url, video_file, quiet=False)

frac = 0.65 
display(Video(data=video_file, embed = True, height=int(720*frac), width=int(1280*frac)))

#loading a YOLO model 
model = YOLO('yolov8x.pt')

#geting names from classes
dict_classes = model.model.names

# Auxiliary functions
def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

### Configurations
#Verbose during prediction
verbose = False
# Scaling percentage of original frame
scale_percent = 50


#-------------------------------------------------------
# Reading video with cv2
video = cv2.VideoCapture(video_file)

# Objects to detect Yolo
class_IDS = [2, 3, 5, 7] 
# Auxiliary variables
centers_old = {}
centers_new = {}
obj_id = 0 
veiculos_contador_in = dict.fromkeys(class_IDS, 0)
veiculos_contador_out = dict.fromkeys(class_IDS, 0)
end = []
frames_list = []
cy_linha = int((1920*2/3)*scale_percent/100 )
cx_sentido = int(1920 * scale_percent/100) 
offset = int(8 * scale_percent/100 )
contador_in = 0
contador_out = 0
print(f'[INFO] - Verbose during Prediction: {verbose}')


# Original informations of video
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video.get(cv2.CAP_PROP_FPS)
print('[INFO] - Original Dim: ', (width, height))

# Scaling Video for better performance 
if scale_percent != 100:
    print('[INFO] - Scaling change may cause errors in pixels lines ')
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print('[INFO] - Dim Scaled: ', (width, height))
    

#-------------------------------------------------------
### Video output ####
video_name = 'result2.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
VIDEO_CODEC = "MP4V"

output_video = cv2.VideoWriter(tmp_output_path, 
                               cv2.VideoWriter_fourcc(*VIDEO_CODEC), 
                               fps, (width, height))


#-------------------------------------------------------
# Executing Recognition 
for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
    
    # reading frame from video
    _, frame = video.read()
    
    #Applying resizing of read frame
    frame  = risize_frame(frame, scale_percent)
    
    if verbose:
        print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

    # Getting predictions
    y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 'cpu', verbose = False)
    
    # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
    boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
    conf    = y_hat[0].boxes.conf.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy() 
    
    # Storing the above information in a dataframe
    # Extracting box data
    boxes = y_hat[0].boxes

    # Create DataFrame with the required information
    positions_frame = pd.DataFrame({
        'xmin': boxes.xyxy[:, 0].cpu().numpy(),
        'ymin': boxes.xyxy[:, 1].cpu().numpy(),
        'xmax': boxes.xyxy[:, 2].cpu().numpy(),
        'ymax': boxes.xyxy[:, 3].cpu().numpy(),
        'conf': boxes.conf.cpu().numpy(),
        'class': boxes.cls.cpu().numpy()
    })
    
    #Translating the numeric class labels to text
    labels = [dict_classes[i] for i in classes]
    
    # Drawing transition line for in\out vehicles counting 
    cv2.line(frame, (0, cy_linha), (int(4500 * scale_percent/100 ), cy_linha), (255,255,0),8)
    
    # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
        
        # Calculating the center of the bounding-box
        center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
        
        # drawing center and bounding-box of vehicle in the given frame 
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
        cv2.circle(frame, (center_x,center_y), 5,(255,0,0),-1) # center of box
        
        #Drawing above the bounding-box the name of class recognized.
        cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
                    org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=2)
        
        # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset 
        if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
            if  (center_x >= 0) and (center_x <=cx_sentido):
                contador_in +=1
                veiculos_contador_in[category] += 1
            else:
                contador_out += 1
                veiculos_contador_out[category] += 1
    
    #updating the counting type of vehicle 
    contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]
    contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
    
    #drawing the number of vehicles in\out
    cv2.putText(img=frame, text='N. vehicles In', 
                org= (10,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=0.5, color=(255, 255, 0),thickness=1)
    
    cv2.putText(img=frame, text='N. vehicles Out', 
                org= (width-200,30), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 0),thickness=1)
    
    #drawing the counting of type of vehicles in the corners of frame 
    xt = 40
    for txt in range(len(contador_in_plt)):
        xt +=30
        cv2.putText(img=frame, text=contador_in_plt[txt], 
                    org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=0.5, color=(255, 255, 0),thickness=1)
        
        cv2.putText(img=frame, text=contador_out_plt[txt],
                    org= (int(2800 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.5, color=(255, 255, 0),thickness=1)
    
    #drawing the number of vehicles in\out
    cv2.putText(img=frame, text=f'In:{contador_in}', 
                org= (int(width * 0.4), cy_linha + 60),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 0),thickness=1)
    
    cv2.putText(img=frame, text=f'Out:{contador_out}', 
                org= (int(width * 0.4), cy_linha - 40),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 0),thickness=1)

    if verbose:
        print(contador_in, contador_out)
    #Saving frames in a list 
    frames_list.append(frame)
    #saving transformed frames in a output video formaat
    output_video.write(frame)
    
#Releasing the video    
output_video.release()


####  pos processing
# Fixing video output codec to run in the notebook\browser
if os.path.exists(output_path):
    os.remove(output_path)
    
ffmpeg_path = r'C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe'  # Replace with your actual FFmpeg path

# Your existing command for FFmpeg (adjust the paths to your actual files)
subprocess.run([
    ffmpeg_path,
    "-i", tmp_output_path,  # Input file
    "-crf", "18",           # Quality factor
    "-preset", "veryfast",  # Encoding speed
    "-hide_banner", "-loglevel", "error", "-vcodec", "libx264",  # Codec
    output_path              # Output file
])

# Checking samples of processed frames
# for i in [28, 29, 32, 40, 42, 50, 58]:
#     plt.figure(figsize =(14, 10))
#     plt.imshow(frames_list[i])
#     plt.show()

frac = 0.7 
Video(data='rep_result1.mp4', embed=True, height=int(720 * frac), width=int(1280 * frac))
