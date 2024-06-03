The codes developed in Python are attached. On the one hand, the "capture_samples" code for the capture and storage in sequence format of images (frames) of 14 dynamic signs that represent basic expressions of the PSL. Additionally, the code "kreate_keypoints" to create and save the keypoints that represent these dynamic PSL keys. On the other hand, the code "training_testing_model_loss_acc_confusionmatrix-graphs" for training and testing the model based on metrics, such as accuracy and loss graphs, as well as the use of the Confusion Matrix for precision, recall and F1-score. It was possible to obtain an LSTM model with high values ​​of accuracy, precision, recall and F1-score which allows the recognition, classification and translation in real time of dynamic sign sequences that represent 14 basic expressions of the PSL. Which in this case are: "hello", "good morning", "good afternoon", "good evening", "how are you?", "I'm fine", "I'm unwell", "more or less", " excuse me”, “please”, “can you help me”, “what time is it?”, “thank you”, and “goodbye”.

## Main Codes
- capture_samples.py → captures the samples of the 14 dynamic signs of the PSL in this case and places them in the frame_actions folder.
- create_keypoints.py → creates and save the keypoints that represent these dynamic PSL keys.
- model.py → to configuration of the hyperparameters of our neural network model based on LSTM architecture for evaluation of training and testing.
- training_testing_model_loss_acc_confusionmatrix-graphs.py → for training and testing the model based on metrics, such as accuracy and loss graphs, as well as the use of the Confusion Matrix for precision, recall and F1-score.

## Steps to test our neural network model based on LSTM architecture
1. Capture the samples with capture_samples.py
2. Generate the .h5 (keypoints) of each word or phrase with create_keypoints.py
3. Training and testing the model based on metrics with training_testing_model_loss_acc_confusionmatrix-graphs.py
