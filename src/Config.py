class Config:
    def __init__(self, audio_dir_path, num_genres, duration_size, sampling_rate, train_part_size, val_part_size, test_part_size, batch_size, learning_rate, epochs, seed, device):
        self.audio_dir_path = audio_dir_path
        self.num_genres = num_genres
        self.duration_size = duration_size
        self.sampling_rate = sampling_rate
        self.train_part_size = train_part_size
        self.val_part_size = val_part_size
        self.test_part_size = test_part_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.device = device