import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        # label = torch.arange(5).repeat(4)
        label = np.array(label)
        # m_ind = []
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            # print(ind)
            self.m_ind.append(ind)
            # m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            # classes = torch.randperm(len(m_ind))[:n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                # print(len(l))
                # print(pos)
                batch.append(l[pos])
            # print(len(batch))
            # print(batch)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch