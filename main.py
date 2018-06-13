import numpy as np
import data_aggregation as da
import model

data = da.Data()

mod = model.Model(0.000001, 0.00001, init_type=model.InitType.NORMAL, epsilon=0.0001)

def accuracy():
    predictions = (np.argmax(mod.predict(data.test_imgs)[3], axis=0) == data.test_lbls)
    print(np.sum(predictions) / predictions.size)

accuracy()

for i in range(100000):
    imgs, lbls, logits = data.next_batch(size=1000)
    mod.minimize(imgs, logits)
    if i%500 == 0:
        accuracy()

accuracy()