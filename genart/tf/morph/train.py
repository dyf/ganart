import numpy as np
from genart.tf.morph.model import MorphModel
from genart.tf.morph.data import load_data
from tensorflow.keras.optimizers import Adam
import sklearn.model_selection as sk
from tensorflow.keras.callbacks import ModelCheckpoint

np.random.seed(0)

f = "E:/Workspace/genart/morphologies.h5"
output_path = "E:/Workspace/genart/morph_weights.h5"
d = load_data(f)
#d = d[:,:1000,:]
np.random.shuffle(d)

X_train, X_test, _, _ = sk.train_test_split(d,d, test_size=0.2, random_state = 42)
checkpoint_callback = ModelCheckpoint(output_path)

m = MorphModel(d.shape[1:])
m.summary()
m.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
m.fit(X_train, X_train, batch_size=1, epochs=100, 
      #validation_data=(X_test, X_test), 
      callbacks=[checkpoint_callback])