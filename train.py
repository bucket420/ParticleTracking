import deeptrack as dt
import numpy as np
import os

training_images = [dt.LoadImage("training_data\\" + image)()._value / 256 for image in os.listdir("training_data")]
model = dt.models.LodeSTAR(input_shape=(None, None, 3))

for img in training_images:
    train_set =   (
        dt.Value(img)
        >> dt.Add(lambda: np.random.randn() * 0.1)
        >> dt.Gaussian(sigma=lambda:np.random.uniform(0, 0.2))  
        >> dt.Multiply(lambda: np.random.uniform(0.6, 1.2))
    )
    model.fit(
        train_set,
        epochs=30,
        batch_size=8,
    )
    
model.save_weights("models\model\weights")