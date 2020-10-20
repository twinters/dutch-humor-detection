# %%
import datetime

import pandas as pd
import torch
import numpy as np
from pycm import ConfusionMatrix

from src.datasets.word_embeddings import RankSequenceDataset
from torch.utils.data import Dataset, DataLoader
# from src.modules.lstm import SingleNetworkLSTMModule, RankedNetworkLSTMModule
from src.modules.cnn import SingleNetworkCNNModule, RankedNetworkCNNModule
import pytorch_lightning as pl
from src.plot_confusion_matrix import confusion_matrix
import kiwi
from src.kiwi_logger import KiwiLogger


def run(original="data/processed/jokes.json",
        replaced="data/processed/dynamic_template_jokes.json"):
    dataset = RankSequenceDataset(original=original,
                                  replaced=replaced,
                                  embeddings_path="./data/raw/roularta-160.txt")

    # Create splits
    train_size = int(0.7 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(1))

    # Create pytoch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    # Register dataset with kiwi
    current_experiment_id: int = kiwi.create_experiment(datetime.datetime.now().__str__())
    kiwi.register_training_dataset(dataloader=train_dataset,
                                   dataset_location=original,
                                   experiment_id=current_experiment_id)
    kiwi.register_dev_dataset(dataloader=val_dataset,
                              dataset_location=original,
                              experiment_id=current_experiment_id)
    kiwi.register_test_dataset(dataloader=test_dataset,
                               dataset_location=original,
                               experiment_id=current_experiment_id)

    def objective(args):
        # start a run
        with kiwi.start_run(experiment_id=current_experiment_id):
            # register hyperparams
            for key, value in args.items():
                kiwi.log_param(key, value)

            # Define model
            model = RankedNetworkCNNModule(args['learning_rate'], dataset.get_embeddings(),
                                            hidden_dim=args['hidden'], output_labels=2)

            # Train (obviously)
            trainer = pl.Trainer(max_epochs=15, logger=KiwiLogger())
            trainer.fit(model, train_loader, val_loader)

            # Evaluation on held-out test-set
            with torch.no_grad():
                model.eval()
                results = pd.DataFrame(columns=['labels', 'predictions'])
                for batch_idx, batch in enumerate(test_loader):
                    y_hat = model(batch['a'], batch['b'])

                    results: pd.DataFrame = results.append(pd.DataFrame({'labels': batch['label'].flatten(),
                                                                         'predictions': y_hat.detach().argmax(axis=1)}),
                                                           ignore_index=True)
                results.to_csv()

                # With a nice confusion matrix
                confusion_matrix(y_pred=results['predictions'].values,
                                 y_true=results['labels'].values, classes=[0, 1])

                cm = ConfusionMatrix(actual_vector=results['labels'].values,
                                     predict_vector=results['predictions'].values)

                output_test_results = "cm.txt"
                cm.save_stat(output_test_results)

                output_test_predictions_file = "test_predictions.txt"
                np.savetxt(output_test_predictions_file, results['predictions'].values, delimiter=",")

                kiwi.log_metric(key="test_acc", value=cm.Overall_ACC)
                kiwi.log_metric(key="test_f1_micro", value=cm.F1_Micro)
                kiwi.log_metric(key="test_f1_macro", value=cm.F1_Macro)
                kiwi.log_metric(key="test_ci_pm", value=cm.CI95[1] - cm.Overall_ACC)
                kiwi.log_metric(key="test_ci_pm", value=cm.CI95[1] - cm.Overall_ACC)
                kiwi.log_artifact(output_test_predictions_file)
                kiwi.log_artifact(output_test_results + ".pycm")

            return cm.Overall_ACC

    space = {
        'learning_rate': ("range", [1e-3, 1e-1]),
        # 'batch_size': ("choice", [4, 8, 16, 32, 64, 128]),
        'hidden': ("choice", [ 16])
    }

    kiwi.start_experiment(current_experiment_id, hp_space=space, objective=objective, max_evals=10, mode="random")
