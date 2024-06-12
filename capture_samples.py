import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from helpers import create_folder, draw_keypoints, mediapipe_detection, save_frames, there_hand
from constants import FONT, FONT_POS, FONT_SIZE, FRAME_ACTIONS_PATH, ROOT_PATH

def capture_samples(path, margin_frame=3, min_cant_frames=5):
    '''
    ### CAPTURE SAMPLES FOR A WORD
    Receives the save path as a parameter and saves the frames
    
    `path` word folder path \n
    `margin_frame` number of frames that are ignored at the beginning and end \n
    `min_cant_frames` min number of frames for each sample
    '''
    create_folder(path)
    
    cant_sample_exist = len(os.listdir(path))
    count_sample = 0
    count_frame = 0
    frames = []
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            frame = video.read()[1] 
            image, results = mediapipe_detection(frame, holistic_model)
            
            if there_hand(results):
                count_frame += 1
                if count_frame > margin_frame: 
                    cv2.putText(image, 'Capturing...', FONT_POS, FONT, FONT_SIZE, (255, 50, 0))
                    frames.append(np.asarray(frame))
                
            else:
                if len(frames) > min_cant_frames + margin_frame:
                    frames = frames[:-margin_frame]
                    output_folder = os.path.join(path, f"sample_{cant_sample_exist + count_sample + 1}")
                    create_folder(output_folder)
                    save_frames(frames, output_folder)
                    count_sample += 1
                
                frames = []
                count_frame = 0
                cv2.putText(image, 'Ready to capture...', FONT_POS, FONT, FONT_SIZE, (0,220, 100))
                
            draw_keypoints(image, results)
            cv2.imshow(f'Taking samples for "{os.path.basename(path)}"', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    word_name = "hello"  # Change by Word/Frase in PSL (hello, how are you, I'm well,...)
    word_path = os.path.join(ROOT_PATH, FRAME_ACTIONS_PATH, word_name)
    capture_samples(word_path)
