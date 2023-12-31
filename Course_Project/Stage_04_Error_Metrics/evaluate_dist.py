"""

Compute error metrics 

"""

import options
import utils
from trainer import Validator
import os
os.path.abspath(os.curdir)
os.chdir("..")
app_path = os.path.abspath(os.curdir)

if __name__ == "__main__":

    print("=======================================================")
    print("Evaluate distance of 3D Point Cloud generation model.")
    print("=======================================================")

    cfg = options.get_arguments()
    cfg.batchSize = cfg.inputViewN
    # cfg.chunkSize = 50

    RESULTS_PATH = f"{app_path}/CAP5415_Assignments/Course_Project/Stage_04_Error_Metrics/results/{cfg.model}_{cfg.experiment}"

    dataloaders = utils.make_data_fixed(cfg)
    test_dataset = dataloaders[1].dataset


    validator = Validator(cfg, test_dataset) 

    hist = validator.eval_dist()
    hist.to_csv(f"{RESULTS_PATH}_testerror.csv", index=False)
