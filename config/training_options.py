class TrainingOptions:
    def __init__(
        self,
        batch_size: int,
        number_of_epochs: int,
        train_folder: str,
        validation_folder: str,
        runs_folder: str,
        start_epoch: int,
        experiment_name: str,
    ):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name