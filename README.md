This project implements a Long Short-Term Memory (LSTM) Neural Network to predict Formula 1 race lap times based on key race parameters, including fuel load, tire degradation, and track temperature. The model is trained on historical F1 race data and can simulate lap-by-lap performance across different tire compounds (Soft, Medium, and Hard). The objective is to assist race engineers in optimizing race strategies through AI-driven insights.
It is made on flask backend to show graph and compare from the real-world lap time produced. 
The data is from FastF1 API.

![dutch laptime predictions](https://github.com/user-attachments/assets/28781bfc-5385-47b0-9d2a-d50db5942a95)

![dutch visualised](https://github.com/user-attachments/assets/d5be11a7-f3a1-4132-a317-1a24010b0ea8)

You can compare real lap times with predicted lap times 
![dutch comparison](https://github.com/user-attachments/assets/cf0d9818-5c59-456c-82fb-cf422e092bcb)

To install the depencies:
pip install flask fastf1 numpy pandas tensorflow keras
Might also be more such as MeanSquaredError, load_model, LeakyReLU etc.
Also change the directories in ANN models, and FastF1 cache.

Reference:
https://ijsrem.com/download/predicting-lap-times-of-formula-1-races-using-ann/


