import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from keras.models import load_model
from helpers import draw_keypoints, extract_keypoints, format_sentences, get_actions, mediapipe_detection, save_txt, there_hand
from text_to_speech import text_to_speech
from constants import DATA_PATH, FONT, FONT_POS, FONT_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES, MODELS_PATH, MODEL_NAME, ROOT_PATH

def evaluate_model(model, threshold=0.9):
    count_frame = 0
    repe_sent = 1
    kp_sequence, sentence = [], []
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            _, frame = video.read()

            image, results = mediapipe_detection(frame, holistic_model)
            kp_sequence.append(extract_keypoints(results))
            
            if len(kp_sequence) > MAX_LENGTH_FRAMES and there_hand(results):
                count_frame += 1
                
            else:
                if count_frame >= MIN_LENGTH_FRAMES:
                    res = model.predict(np.expand_dims(kp_sequence[-MAX_LENGTH_FRAMES:], axis=0))[0]

                    if res[np.argmax(res)] > threshold:
                        sent = actions[np.argmax(res)]
                        sentence.insert(0, sent)
                        text_to_speech(sent)
                        sentence, repe_sent = format_sentences(sent, sentence, repe_sent)
                        
                    count_frame = 0
                    kp_sequence = []
            
            TEXT_BG_COLOR = (32, 35, 217)  # Background color of the text (in this case, a type of red)
            TEXT_BG_HEIGHT = 40  # Height of text background rectangle

            # Draw a background rectangle for the text
            cv2.rectangle(image, (0, 0), (640, TEXT_BG_HEIGHT), TEXT_BG_COLOR, -1)

            # Draw text on the background
            cv2.putText(image, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
            save_txt('outputs/sentences.txt', '\n'.join(sentence))
            
            draw_keypoints(image, results)
            cv2.imshow('Peruvian Sign Language (PSL) real-time continuous translator', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
        video.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    actions = get_actions(DATA_PATH)
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    lstm_model = load_model(model_path)
    
    evaluate_model(lstm_model)






