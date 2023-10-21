'''

__author__ = 'mmms'

-> read videos with opencv and get key points
in the frames(face,pose,hands)

'''

#import dependencies

import  numpy as np        # !pip install numpy
import cv2                 # !pip install opencv-python
import mediapipe as mp     # !pip install mediapipe
import time
import os
import sklearn# !pip install sklearn
import matplotlib.pyplot as plt    # !pip install matplotlib
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM,Dense

#declaration of variables and data paths
DATA_PATH = r'C:\Users\mmms\Desktop\project\random_file_lib\Sema\DATA'
MODEL_PATH = r'C:\Users\mmms\Desktop\project\random_file_lib\Sema\sign_lang_mdl.h5'
cap = cv2.VideoCapture(0)
training_model = mp.solutions.holistic
training_drawing = mp.solutions.drawing_utils
actions = ['hello','iloveyou','thankyou']
sequences = 30
sequence_len = 30
label_map = {label:num for num in enumerate(actions)}
training_permission = False
if input('Train a model? [y/n]  ').lower() == 'y':
    training_permission = True
    while True:
        model_name = input('What operation do you wish to train ?  ["q" to quit]  ').lower()
        if model_name != "q":
            if model_name not in r"os.path.listdir(DATA_PATH)":
                actions.append(model_name)
        else:
            break

    if input('Show LandMarks? y/n   ').lower() == 'y':
        show_landmark = True
    else:
        show_landmark = False


class ModelTrain():

    def __init__(self,X_data:None,y_data:None):
        self.X_data = X_data
        self.y_data = y_data
        self.new_model = new_model

    def start_detection(self):
        sequence = []
        sentence = []
        threshold = 0.8
        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                sequence.insert(0,keypoints)
                sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = MODEL_PATH.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


    def handle_data(self):
        X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.05)
        lang_model = Sequential()
        lang_model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
        lang_model.add(LSTM(128,return_sequences=True,activation='relu'))
        lang_model.add(LSTM(64,return_sequences=True,activation='relu'))
        lang_model.add(Dense(64,activation='relu'))
        lang_model.add(Dense(32,activation='relu'))
        lang_model.add(Dense(actions.shape[0],activation='softmax'))

        lang_model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
        model.fit(X_train,y_train,epochs=2000,verbose=0)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        lang_model.save(MODEL_PATH)

    def get_data(self):
        sequences, labels = [], []
        for actn in actions:
            for seq in range(no_sequences):
                window = []
                for fram in range(sequence_len):
                    res = np.load(os.path.join(DATA_PATH, action, str(seq), "{}.npy".format(fram)))
                    window.append(res)
                    sequences.append(window)
                    labels.append({label: num for num, label in enumerate(actions)}[actn])
        return np.array(sequences),to_categorical(labels).astype(int)

class ModelSema:
    def __init__(self,frame,model,res):
        self.frame = frame
        self.model = model
        self.res = res


    def process_mdl(self):
        try:
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            processed = self.model.process(img)
            img.flags.writeable = True
            img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            return img,processed
        except cv2.error:
            pass

    def make_directory(self):
        for action in actions:
            for sequence in range(sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except:
                    pass

    def draw_connections(self):
        training_drawing.draw_landmarks(self.frame,self.res.face_landmarks,training_model.FACEMESH_CONTOURS,
                                        training_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),
                                        training_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)
                                        )
        training_drawing.draw_landmarks(self.frame, self.res.pose_landmarks, training_model.POSE_CONNECTIONS,
                                        training_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                        training_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                        )
        training_drawing.draw_landmarks(self.frame, self.res.left_hand_landmarks, training_model.HAND_CONNECTIONS,
                                        training_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                        training_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                        )
        training_drawing.draw_landmarks(self.frame, self.res.right_hand_landmarks, training_model.HAND_CONNECTIONS,
                                        training_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                        training_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                        )
    def extract_keypoints(self):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3),
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])



    def collect_keypoints(self):
        pose_lmk = np.array([[r.x, r.y, r.z, r.visibility] for r in self.res.pose_landmarks.landmark]).flatten() if \
            self.res.pose_landmarks else np.zeros(33 * 4)
        face_lmk = np.array([[r.x, r.y, r.z] for r in self.res.face_landmarks.landmark]) if \
            self.res.pose_landmarks else np.zeros(468 * 3)
        lh_lmk = np.array([[r.x, r.y, r.z] for r in self.res.left_hand_landmarks.landmark]) if \
            self.res.pose_landmarks else np.zeros(21 * 3)
        rh_lmk = np.array([[r.x, r.y, r.z] for r in self.res.right_hand_landmarks.landmark]) if \
            self.res.pose_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose_lmk,face_lmk,lh_lmk,rh_lmk])


obj2 = ModelTrain()
X,y = obj2.get_data()
obj3 = ModelTrain(X,y)
obj3.handle_data()
if cap.isOpened():
    with training_model.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(sequences):
                for frame_num in range(sequence_len):
                    ret, frame = cap.read()
                    try:
                        img, processed = ModelSema(frame, holistic, True).process_mdl()
                    except TypeError:
                        break
                    obj1 = ModelSema(img, holistic, processed)
                    if not os.path.exists(DATA_PATH):
                        obj1.make_directory()
                    if show_landmark:
                        obj1.draw_connections()
                    if frame_num == 0:
                        cv2.putText(img, 'STARTING COLLECTION', (35,50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 0), 2, cv2.LINE_AA)
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', img)
                        cv2.waitKey(3000)
                    else:
                        cv2.putText(img, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', img)

                    keypoints = obj1.collect_keypoints()
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0XFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()


print('Training ... ')
obj2 = ModelTrain()
X,y = obj2.get_data()
obj3 = ModelTrain(X,y)
obj3.handle_data()
