"""Defines the configuration to be loaded before running any experiment"""
import string
from configobj import ConfigObj


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config


        # Model name and location to store
        self.model_name = config["train"]["model_name"]
        self.log_dir = config["train"]["log_dir"]
        self.log_interval = config["train"].as_int("log_interval")
        self.relation_loss_weight = config["train"].as_float("relation_loss_weight")
        self.ms_iter = config["train"].as_int("ms_iter")

        # positional encoding
        self.position_encode = config["train"].as_bool("position_encode")
        self.encode_level = config["train"].as_int("encode_level")

        self.preload_model = config["train"].as_bool("preload_model")
        # path to the models
        self.pretrain_model_path = config["train"]["pretrain_model_path"]
        self.pretrain_model_name = config["train"]["pretrain_model_name"]

        # Normals
        self.normals = config["train"].as_bool("normals")

        # number of training examples
        self.num_train = config["train"].as_int("num_train")
        self.num_val = config["train"].as_int("num_val")
        self.num_test = config["train"].as_int("num_test")
        self.num_points = config["train"].as_int("num_points")
        self.grid_size = config["train"].as_int("grid_size")
        # Weight to the loss function for stretching
        self.loss_weight = config["train"].as_float("loss_weight")

        # dataset
        self.dataset_path = config["train"]["dataset"]
        self.train_data = config["train"]["train_list"]
        self.test_data = config["train"]["test_list"]
        self.augment = config["train"].as_int("augment")
        self.if_normal_noise = config["train"].as_int("if_normal_noise")
        self.train_skip = config["train"].as_int("train_skip")
        self.val_skip = config["train"].as_int("val_skip")

        # Proportion of train dataset to use
        self.proportion = config["train"].as_float("proportion")

        # Number of epochs to run during training
        self.epochs = config["train"].as_int("num_epochs")

        # batch size, based on the GPU memory
        self.batch_size = config["train"].as_int("batch_size")

        # Mode of training, 1: supervised, 2: RL
        self.mode = config["train"].as_int("mode")

        # Learning rate
        self.lr = config["train"].as_float("lr")

        # Number of epochs to wait before decaying the learning rate.
        self.patience = config["train"].as_int("patience")

        # Optimizer: RL training -> "sgd" or supervised training -> "adam"
        self.optim = config["train"]["optim"]

        # Epsilon for the RL training, not applicable in Supervised training
        self.accum = config["train"].as_int("accum")

        # Whether to schedule the learning rate or not
        self.lr_sch = config["train"].as_bool("lr_sch")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())
