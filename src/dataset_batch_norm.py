from torch.utils.data import Dataset

class BatchData(Dataset):
    def __init__(self, lr_sound, hr_sound, lr_mean, lr_std, hr_mean, hr_std):
        self.lr_sound = lr_sound
        self.hr_sound = hr_sound
        self.lr_mean = lr_mean
        self.lr_std = lr_std
        self.hr_mean = hr_mean
        self.hr_std = hr_std

    def __getitem__(self, index):
        lr_sound = self.lr_sound[index]
        hr_sound = self.hr_sound[index]

        lr_sound = 1*(lr_sound - self.lr_mean)/self.lr_std
        hr_sound = 1*(hr_sound - self.hr_mean) / self.hr_std
        return lr_sound, hr_sound

    def __len__(self):
        return len(self.lr_sound)