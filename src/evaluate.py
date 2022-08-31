"Collect best metrics from experiment"
import json
import numpy as np
from pathlib import Path

# time per epoch
times = []
with open("data/train/time_log.csv") as fp:
    for j,line in enumerate(fp.readlines()):
        # not consider header
        if j>0:
            epoch, time = line.split("\t")
            times.append(float(time))
avg_time_epoch=sum(times)/len(times)

# best epoch
best_epoch = 0
best_val_loss = np.inf
with open("data/train/training_log.csv") as fp:
    for j,line in enumerate(fp.readlines()):
        # not consider header
        if j>0:
            epoch,loss,lr,val_loss = line.split("\t")
            
            if float(val_loss) < best_val_loss:
                best_loss=float(loss)
                best_val_loss = float(val_loss)
                best_epoch = float(epoch)+1 # epochs are indexed from 0

# save metrics
metrics={
    "best_epoch": best_epoch,
    "loss": best_loss,
    "val_loss": best_val_loss,
    "avg_time_epoch": avg_time_epoch 
}

PATH_EVAL=Path("data/eval")
PATH_EVAL.mkdir(exist_ok=True, parents=True)
with open(PATH_EVAL.joinpath("metrics.json"),"w") as fp:
    json.dump(metrics,fp, indent=4)