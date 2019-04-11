"""
因为系统自带的train没有visdom功能，这里进行 重载
+ 添加一个入口参数 ` vis: visdom.Visdom = None`
+ 添加一行`trainer.visdom=vis`
"""

import logging
import os
from allennlp.common.checks import check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging, cleanup_global_logging, dump_metrics
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.util import create_serialization_dir, evaluate
import visdom

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def train_model(params: Params,
                serialization_dir: str,
                file_friendly_logging: bool = False,
                recover: bool = False,
                force: bool = False,
                vis: visdom.Visdom = None) -> Model:
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    force : ``bool``, optional (default=False)
        If ``True``, we will overwrite the serialization directory if it already exists.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)
    create_serialization_dir(params, serialization_dir, recover, force)
    stdout_handler = prepare_global_logging(serialization_dir, file_friendly_logging)

    cuda_device = params.params.get('trainer').get('cuda_device', -1)
    check_for_gpu(cuda_device)

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    evaluate_on_test = params.pop_bool("evaluate_on_test", False)

    trainer_type = params.get("trainer", {}).get("type", "default")

    if trainer_type == "default":
        # Special logic to instantiate backward-compatible trainer.
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        trainer = Trainer.from_params(
                model=pieces.model,
                serialization_dir=serialization_dir,
                iterator=pieces.iterator,
                train_data=pieces.train_dataset,
                validation_data=pieces.validation_dataset,
                params=pieces.params,
                validation_iterator=pieces.validation_iterator)
        evaluation_iterator = pieces.validation_iterator or pieces.iterator
        evaluation_dataset = pieces.test_dataset

    else:
        trainer = TrainerBase.from_params(params, serialization_dir, recover)
        # TODO(joelgrus): handle evaluation in the general case
        evaluation_iterator = evaluation_dataset = None

    params.assert_empty('base train command')
    
    #########################################
    trainer.visdom=vis
    #########################################
    
    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise
    
    # Evaluate
    if evaluation_dataset and evaluate_on_test:
        logger.info("The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(trainer.model, evaluation_dataset, evaluation_iterator,
                                cuda_device=trainer._cuda_devices[0], # pylint: disable=protected-access,
                                # TODO(brendanr): Pass in an arg following Joel's trainer refactor.
                                batch_weight_key="")

        for key, value in test_metrics.items():
            metrics["test_" + key] = value

    elif evaluation_dataset:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    cleanup_global_logging(stdout_handler)

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)
    dump_metrics(os.path.join(serialization_dir, "metrics.json"), metrics, log=True)

    # We count on the trainer to have the model with best weights
    return trainer.model