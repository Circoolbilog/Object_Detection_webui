import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2


class Config:
    def __init__(self,
                 pipeline_config_path=None,
                 num_train_steps=None,
                 eval_on_train_data=False,
                 sample_1_of_n_eval_examples=None,
                 sample_1_of_n_eval_on_train_examples=5,
                 model_dir=None,
                 checkpoint_dir=None,
                 eval_timeout=3600,
                 use_tpu=False,
                 tpu_name=None,
                 num_workers=1,
                 checkpoint_every_n=1000,
                 record_summaries=True):
        self.pipeline_config_path = pipeline_config_path
        self.num_train_steps = num_train_steps
        self.eval_on_train_data = eval_on_train_data
        self.sample_1_of_n_eval_examples = sample_1_of_n_eval_examples
        self.sample_1_of_n_eval_on_train_examples = sample_1_of_n_eval_on_train_examples
        self.model_dir = model_dir
        self.checkponit_dir = checkpoint_dir
        self.eval_timeout = eval_timeout
        self.use_tpu = use_tpu
        self.tpu_name = tpu_name
        self.num_workers = num_workers
        self.checkponit_every_n = checkpoint_every_n
        self.record_summaries = record_summaries


def run_model(config):
    if not config.model_dir or not config.pipeline_config_path:
        raise ValueError("model_dir and pipeline_config_path are required")

    tf.config.set_soft_device_placement(True)

    if config.checkpoint_dir:
        model_lib_v2.eval_continuously(
            pipeline_config_path=config.pipeline_config_path,
            model_dir=config.model_dir,
            train_steps=config.num_train_steps,
            sample_1_of_n_eval_examples=config.sample_1_of_n_eval_examples,
            sample_1_of_n_eval_on_train_examples=config.sample_1_of_n_eval_on_train_examples,
            checkpoint_dir=config.checkpoint_dir,
            wait_interval=300,
            timeout=config.eval_timeout)

    else:
        # if config.use_tpu:
        # elif config.num_workers > 1:
        if config.num_weorkers > 1:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.compat.v2.distribute.MirroredStrategy()

        with strategy. scope():
            model_lib_v2.train_loop(
                pipeline_config_path=config.pipeline_config_path,
                model_dir=config.model_dir,
                train_steps=config.num_train_steps,
                use_tpu=config.use_tpu,
                checkpoint_every_n=config.checkpoint_every_n,
                record_summaries=config.record_summaries)
