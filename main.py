import numpy as np
import data_aggregation as da
import model

data = da.Data()

mod = model.Model(0.85, 0.0000002, init_type=model.InitType.NORMAL, epsilon=0.1)

def accuracy():
    predictions = (np.argmax(mod.predict(data.test_imgs)[3], axis=0) == data.test_lbls)
    print(np.sum(predictions) / predictions.size)

accuracy()

for i in range(100000):
    imgs, lbls, logits = data.next_batch(size=200)
    mod.minimize(imgs, logits)
    if i%50 == 0:
        accuracy()

accuracy()