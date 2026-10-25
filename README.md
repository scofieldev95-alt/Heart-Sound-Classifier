# Heart-Sound-Classifier
This project screens heart audio for abnormalities using Random Forest and MFCC features. Trained on mixed data, it scores 74% accuracy. However, evaluation exposes the "Stethoscope-to-Phone Gap": a model trained strictly on clinical stethoscopes drops to 37% accuracy and 0% recall on mobile recordings, proving domain adaptation is vital.
