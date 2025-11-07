import os
from sklearn.model_selection import train_test_split

def get_partitioned_data(config):
    label_map={'blues' : 0, 'classical' : 1, 'country' : 2,
           'disco' : 3, 'hiphop'    : 4, 'jazz'    : 5,
           'metal' : 6, 'pop'       : 7, 'reggae'  : 8, 'rock' : 9}

    # Automatically select the first `num_genres` genres from label_map
    selected_genres = list(label_map.keys())[:config.num_genres]

    # Build a reduced label map only for selected genres
    label_map = {genre: i for i, genre in enumerate(selected_genres)}

    print(f"Using {config.num_genres} genres: {selected_genres}")
    print("Updated label map:", label_map)


    audio_files = []
    labels = []

    # Then make sure your folder traversal loop only collects those folders:
    for genre in os.listdir(config.audio_dir_path):
        if genre not in label_map:
            continue
        genre_path = os.path.join(config.audio_dir_path, genre)
        for fname in os.listdir(genre_path):
            if fname.endswith('.wav'):
                file_path = os.path.join(genre_path, fname)
                audio_files.append(file_path)
                labels.append(label_map[genre])

    print(f"Total selected files: {len(audio_files)}")        

    # ration training - validation - test data
    # 70% Training, 15% Validation, 15% Test


    # 70% train, 30% temp
    pseudo_ts = config.test_part_size + config.val_part_size
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        audio_files, labels, test_size=pseudo_ts, stratify=labels, random_state=config.seed
    )

    # Split temp 30% -> 15% val + 15% test
    pseudo_ts = config.test_part_size / config.test_part_size + config.val_part_size
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config.seed
    )

    print(f"Training set: {len(train_files)}")
    print(f"Validation set: {len(val_files)}")
    print(f"Test set: {len(test_files)}")

    assert len(set(train_files) & set(val_files) & set(test_files)) == 0

    return train_files, train_labels, val_files, val_labels, test_files, test_labels