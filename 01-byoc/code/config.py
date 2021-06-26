class ParameterSetting():
    def __init__(self, csv_path='./', data_dir='furbo_only', save_root='snapshots', model_file='snapshots/final_model.pkl', 
                 model_name = 'CNN14', val_split=0,
                 epochs=20, batch_size=128, lr=0.0001, num_class=2,
                 time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2,
                 sr=8000, nfft=200, hop=80, mel=64, resume=None, normalize=None, preload=False,
                 spec_aug=False, optimizer='adam', scheduler='cosine'):

        self.csv_path = csv_path
        self.data_dir = data_dir
        self.save_root = save_root
        self.model_file = model_file 
        self.model_name = model_name
        self.val_split = val_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_class = num_class

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

        self.sr = sr
        self.nfft = nfft
        self.hop = hop
        self.mel = mel

        self.resume = resume
        self.normalize = normalize
        self.preload = preload
        self.spec_aug = spec_aug
