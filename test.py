import numpy as np
import data_aggregation as da

data = da.Data()

imgs, lbls, lbl_logits = data.next_batch(size=100)

index = 0

print(lbls[index])
print(np.argmax(np.take(lbl_logits, (index), axis=1)))
data.display_image(np.take(imgs, (index), axis=1))