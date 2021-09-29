# Trafic-Light-for-Life (SAS Hackathon 2021)

A data for good project accepted for participation in Global HackinSAS 2021. Below are the highlights of the project, followed by technical details and model execution instructions:
- __The central idea:__ smart intersection management through audio sensing to prioritize emergency vehicles (please see the submission here: https://communities.sas.com/t5/Hacker-s-Hub-library/Traffic-Lights-for-Life-audio-based-intersection-management-for/ta-p/716595)
- __Raspberry Pi Project:__ Bi-directional set-up hosted by Azure IoT Hub
- __Artificial Intelligence:__ Application of CNN & LSTM (with 5 sec Audio Chunks) & CNN (with Spectrograms) for binary classification of emergency vs. non-emergency vehicles
- __Validation Dataset:__ 4:00 minutes of audio data including both emeregncy and non-emergency vehicles, as shown below: <br> 
     Audio: https://github.com/amirhossini/Trafic-Light-for-Life/blob/main/data/validation.wav <br>
     Time-series: <br> ![image](https://user-images.githubusercontent.com/63076229/109926219-f222f480-7c7f-11eb-8c2b-96154b5de0d8.png) <br>
     Spectrogram: <br> ![image](https://user-images.githubusercontent.com/63076229/112779376-dbb05480-9003-11eb-91ea-61ae9486e4f7.png) <br>

- __Models:__
     - CNN with time-series data: <br> ![image](https://user-images.githubusercontent.com/63076229/112785556-79f6e700-9011-11eb-9681-b97a5b4f152d.png) <br>
     - LSTM with time-series data: <br> ![image](https://user-images.githubusercontent.com/63076229/112785639-b0346680-9011-11eb-9fd2-2d185725e94b.png) <br> 
     - CNN with spectrogram data: <br> ![image](https://user-images.githubusercontent.com/63076229/112785689-d4904300-9011-11eb-8afd-9cd1230f938c.png) <br>
     
- __Tunning Performance:__
     - Training Speed: <br> ![image](https://user-images.githubusercontent.com/63076229/112786029-96dfea00-9012-11eb-8042-d2ce5698722b.png) <br>
     - Prediction Speed: <br> ![image](https://user-images.githubusercontent.com/63076229/112786077-b0813180-9012-11eb-8d38-2b893f1efcda.png) <br>
     - Prediction Accuracy on the testing set: <br> ![image](https://user-images.githubusercontent.com/63076229/112786139-d1498700-9012-11eb-96b1-a8ba369619f9.png) <br>

- __Performance on the Validation Set:__ 
     - CNN with time series data outperformaing other techniques <br> ![image](https://user-images.githubusercontent.com/63076229/112786410-72384200-9013-11eb-9ede-c6edc23124c2.png) <br>

- __Model Execution:__ 
     - Please use ./notebooks/data_prep.ipynb for .WAV data sampling, and training/testing/validation set creation
     - Please use ./notebooks/ML_Models.ipynb for ML model architecture design and hyperparameter tuning
     - Best model artefacts are stored under: ./model_artefacts





