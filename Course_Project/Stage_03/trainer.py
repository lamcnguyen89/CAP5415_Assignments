"""

This module creates the training setup, configuration and validation

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch

import transform


# Tests the accuracy of the Autoencoder in its task of creating a 3D point cloud model from a single 2D image
class Validator:
    '''Perform Validation on the trained Structure generator'''
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.device = cfg.device
        self.dataset = dataset
        self.history = []
        self.CADs = dataset.CADs
        self.result_path = f"results/{cfg.model}_{cfg.experiment}"

    def eval(self, model):
        print("======= EVALUATION START =======")

        fuseTrans = self.cfg.fuseTrans
        for i in range(len(self.dataset)):
            cad = self.dataset[i]
            input_images = torch.from_numpy(cad['image_in'])\
                                .permute((0,3,1,2))\
                                .float().to(self.cfg.device)
            points24 = np.zeros([self.cfg.inputViewN, 1], dtype=np.object)

            XYZ, maskLogit = model(input_images)
            mask = (maskLogit > 0).float()
            # ------ build transformer ------
            XYZid, ML = transform.fuse3D(
                self.cfg, XYZ, maskLogit, fuseTrans) # [B,3,VHW],[B,1,VHW]
            
            XYZid, ML = XYZid.permute([0, 2, 1]), ML.squeeze()
            for a in range(self.cfg.inputViewN):
                xyz = XYZid[a] #[VHW, 3]
                ml = ML[a] #[VHW]
                points24[a, 0] = (xyz[ml > 0]).detach().cpu().numpy()

            pointMeanN = np.array([len(p) for p in points24[:, 0]]).mean()
            scipy.io.savemat(
                f"{self.result_path}/{self.CADs[i]}.mat", 
                {"image": cad["image_in"], "pointcloud": points24})

            print(f"{pointMeanN:.2f} points save to {self.result_path}/{self.CADs[i]}.mat")
            self.history.append(
                {"cad": self.CADs[i], "average points": pointMeanN})

        print("======= EVALUATION DONE =======")
        return pd.DataFrame(self.history)

    def eval_dist(self):
        print("======= EVALUATION START =======")
        CADN = len(self.CADs)

        pred2GT_all = np.ones([CADN, self.cfg.inputViewN]) * np.inf
        GT2pred_all = np.ones([CADN, self.cfg.inputViewN]) * np.inf
        with torch.set_grad_enabled(False):
            for m, cad in enumerate(self.CADs):
                # load GT
                obj = scipy.io.loadmat(f"{self.cfg.path}/{self.cfg.category}_testGT/{cad}.mat")
                Vgt = torch.from_numpy(np.concatenate([obj["V"], obj["Vd"]], axis=0)).to(self.device).float()
                VgtN = len(Vgt)
                # load prediction
                Vpred24 = scipy.io.loadmat(f"{self.result_path}/{cad}.mat")["pointcloud"][:, 0]
                assert (len(Vpred24) == self.cfg.inputViewN)

                for a in range(self.cfg.inputViewN):
                    Vpred = torch.from_numpy(Vpred24[a]).to(self.device).float()
                    VpredN = len(Vpred)
                    # rotate CAD model to be in consistent coordinates
                    Vpred[:, 1], Vpred[:, 2] = Vpred[:, 2], -Vpred[:, 1]
                    # compute test error in both directions
                    pred2GT_all[m, a] = self._computeTestError(Vpred, Vgt, type="pred->GT")
                    GT2pred_all[m, a] = self._computeTestError(Vgt, Vpred, type="GT->pred")

                info = {"cad": cad,
                        "pred->GT": pred2GT_all[m].mean()*100,
                        "GT->pred": GT2pred_all[m].mean()*100,}
                print(info)
                self.history.append(info)

        print("======= EVALUATION DONE =======")
        return pd.DataFrame(self.history)

    def _computeTestError(self, Vs, Vt, type):
        """compute test error for one prediction"""
        VsN, VtN = len(Vs), len(Vt)
        if type == "pred->GT":
            evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 100000
        if type == "GT->pred":
            evalN, VsBatchSize, VtBatchSize = min(VsN, 200), 200, 40000
        # randomly sample 3D points to evaluate (for speed)
        randIdx = np.random.permutation(VsN)[:evalN]
        Vs_eval = Vs[randIdx]
        minDist_eval = np.ones([evalN]) * np.inf
        # for batches of source vertices
        VsBatchN = int(np.ceil(evalN / VsBatchSize))
        VtBatchN = int(np.ceil(VtN / VtBatchSize))
        for b in range(VsBatchN):
            VsBatch = Vs_eval[b * VsBatchSize:(b + 1) * VsBatchSize]
            minDist_batch = np.ones([len(VsBatch)]) * np.inf
            for b2 in range(VtBatchN):
                VtBatch = Vt[b2 * VtBatchSize:(b2 + 1) * VtBatchSize]
                _, minDist = self._projection(VsBatch, VtBatch)
                minDist = minDist.detach().cpu().numpy()
                minDist_batch = np.minimum(minDist_batch, minDist)
            minDist_eval[b * VsBatchSize:(b + 1) * VsBatchSize] = minDist_batch
        return np.mean(minDist_eval)

    def _projection(self, Vs, Vt):
        '''compute projection from source to target'''
        VsN = Vs.size(0)
        VtN = Vt.size(0)
        diff = Vt[None, :, :] - Vs[:, None, :]
        dist = (diff**2).sum(dim=2).sqrt()
        idx = torch.argmin(dist, dim=1)
        # proj = Vt_rep[np.arange(VsN), idx, :]
        proj = None
        minDist = dist[np.arange(VsN), idx]

        return proj, minDist
