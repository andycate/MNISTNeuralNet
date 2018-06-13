import numpy as np
import data_aggregation as da
import model

data = da.Data()

mod = model.Model(0.000001, 0.0000001)

def accuracy():
    predictions = (np.argmax(mod.predict(data.test_imgs), axis=0) == data.test_lbls)
    print(np.sum(predictions) / predictions.size)

accuracy()

for i in range(100):
    imgs, lbls, logits = data.next_batch(size=100)
    mod.minimize(imgs, logits)
    if i%50 == 0:
        accuracy()

accuracy()