import deeptrack as dt
import numpy as np
import os

img = dt.LoadImage("training_data\\sample.png")()._value[:, :, :3] / 256
model = dt.models.LodeSTAR(input_shape=(None, None, 3))

train_set =   (
    dt.Value(img)
    >> dt.Add(lambda: np.random.randn() * 0.1)
    >> dt.Gaussian(sigma=lambda:np.random.uniform(0, 0.2))  
    >> dt.Multiply(lambda: np.random.uniform(0.6, 1.2))
)
model.fit(
    train_set,
    epochs=100,
    batch_size=8,
)

model.save_weights("models\model\weights")