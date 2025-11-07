class Config:
    def __init__(self, audio_dir_path, num_genres, train_part_size, val_part_size, test_part_size, batch_size, learning_rate, epochs, seed):
        self.audio_dir_path = audio_dir_path
        self.num_genres = num_genres
        self.train_part_size = train_part_size
        self.val_part_size = val_part_size
        self.test_part_size = test_part_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed