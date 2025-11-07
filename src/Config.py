class Config:
    def __init__(self, audio_dir_path, batch_size, epochs, seed):
        self.audio_dir_path = audio_dir_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed