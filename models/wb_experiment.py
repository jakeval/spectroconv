import json
import os
import pickle
import enum
from collections import namedtuple

#from inspect import Parameter
#import torchsummary
import torch
from torch import nn
from sklearn.metrics import f1_score
import wandb
from tqdm.auto import tqdm
import numpy as np

from data_utils import nsynth_adapter as na
from models import cnn_model, lc_model, lrlc_model


def _as_named_tuple(parameters):
    return namedtuple("run_parameters", parameters.keys())(*parameters.values())


class ModelType(enum.IntEnum):
    CNN = 0
    LC = 1
    LRLC = 2


class WBExperiment:
    def __init__(self, wb_config, wb_defaults='./.wb.config', login_key=False, device=None, storage_dir='./.wandb_store'):
        if login_key:
            with open('./.wandb.key') as f:
                self.token = f.read().strip()
            wandb.login(key=self.token)
        else:
            wandb.login()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if isinstance(wb_defaults, str) and os.path.exists(wb_defaults):
            with open(wb_defaults) as f:
                self.wb_config = json.load(f)
        elif isinstance(wb_defaults, dict):
            self.wb_config = wb_defaults
        else:
            self.wb_config = {}
        self.wb_config.update(wb_config)
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.mkdir(storage_dir)

    def metric_accumulator(self):
        metrics = {
            'accuracy': 0,
            'f1': 0
        }
        count = 0

        def accumulate(y_true, y_pred):
            nonlocal metrics, count

            if torch.cuda.is_available():
                y_pred = y_pred.cpu()
                y_true = y_true.cpu()

            accuracy = y_pred.eq(y_true.view_as(y_pred)).sum()
            f1 = f1_score(y_true, y_pred, average='macro') * y_true.shape[0]

            metrics['accuracy'] += accuracy
            metrics['f1'] += f1
            count += y_true.shape[0]
            return metrics

        def finalize():
            return dict([(k, v / count) for k, v in metrics.items()])

        return accumulate, finalize

    def validate_model(self, model, data_loader):
        model.eval()
        loss = 0.0
        accumulate, finalize = self.metric_accumulator()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                scores = model(X)
                loss += model.loss(scores, y, reduction='sum')  # sum up batch loss
                pred = scores.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                accumulate(y, pred)
        
        metrics = finalize()
        metrics['loss'] = loss / len(data_loader.dataset)
        model.train()
        return metrics

    def log_metrics(self, model, split, data, run, parameters, epoch=None):
        data_loader = data.get_dataloader(parameters.batch_size, shuffle=False)
        metrics = self.validate_model(model, data_loader)

        loss = metrics['loss']
        accuracy = metrics['accuracy']
        f1 = metrics['f1']
        if epoch == None:
            print(f'Split: {split}, Loss: {loss:.2f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
        else:
            print(f'Split: {split}, Epoch: {epoch}, Loss: {loss:.2f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
        
        metrics['split'] = split
        metrics['epoch'] = epoch
        run.log(metrics)

    def get_data(self, config, run):
        data = {}
        code_lookup = None

        for split, split_config in config.items():
            name = split_config['name']
            version = split_config.get('version', 'latest')
            dataset_artifact = run.use_artifact(f'{name}:{version}')
            source = dataset_artifact.metadata['dataset_url']
            nds = na.NsynthDataset(source=source)
            if code_lookup is None:
                code_lookup = nds.initialize()
            else:
                nds.initialize(code_lookup)
            data[split] = nds
        return data

    def update_data(self, data, class_names):
        code_lookup = dict([(e.value, i) for i, e in enumerate(class_names)])
        for split in data.keys():
            data[split].set_code_lookup(code_lookup)
        return data

    def get_model(self, model_type, input_shape, class_enums, parameters):
        #return cnn_model.OriginalCNN(input_shape, class_enums).float().to(self.device)
        if model_type == ModelType.CNN:
            return cnn_model.CnnClf(input_shape, class_enums, parameters).float().to(self.device)
        if model_type == ModelType.LC:
            return lc_model.LcClf(input_shape, class_enums, parameters).float().to(self.device)
        if model_type == ModelType.LRLC:
            return lrlc_model.LrlcClf(input_shape, class_enums, parameters).float().to(self.device)

    def load_model(self, model_name, run, input_shape, model_version='latest'):
        model_artifact = run.use_artifact(f"{model_name}:{model_version}")
        model_config = model_artifact.metadata
        model_dir = model_artifact.download(root=self.storage_dir)
        model_path = os.path.join(model_dir, model_config['model_path'])
        class_names_path = os.path.join(model_dir, model_config['class_names_path'])
        class_enums = None
        with open(class_names_path, 'rb') as f:
            class_enums = pickle.load(f)
        model_type = model_config['id']
        run_parameters = _as_named_tuple(model_config['parameters'])
        model = self.get_model(model_type, input_shape, class_enums, run_parameters)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        return model

    def save_model(self, model, model_config, run_parameters_dict, run):
        model_name = model_config["name"]
        model_filename = f"{model_name}.pth"
        class_names_filename = f"{model_name}.pkl"
        model_file = os.path.join(self.storage_dir, model_filename)
        class_names_file = os.path.join(self.storage_dir, class_names_filename)
        metadata = dict(model_config)
        metadata['parameters'] = run_parameters_dict
        metadata.update({
            'model_path': model_filename,
            'class_names_path': class_names_filename
        })
        params = {
            "name": model_name,
            "type": 'model',
            "metadata": metadata
        }
        model_artifact = wandb.Artifact(**params)
        torch.save(model.state_dict(), model_file)
        model_artifact.add_file(model_file)
        wandb.save(model_file)
        with open(class_names_file, 'wb') as f:
            pickle.dump(model.class_names, f)
        model_artifact.add_file(class_names_file)
        wandb.save(class_names_file)
        run.log_artifact(model_artifact)
        return model_name

    def get_optimizer(self, parameters, model):
      # return torch.optim.SGD(model.parameters(), lr=parameters.learning_rate, momentum=0.9)
      if parameters.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=parameters.learning_rate, momentum=parameters.momentum, weight_decay=parameters.l2_reg)
      elif parameters.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=parameters.learning_rate, weight_decay=parameters.l2_reg)


class SweepExperiment(WBExperiment):
    def __init__(self, wb_config, wb_defaults='./.wb.config', login_key=False, device=None, storage_dir='./.wandb_store'):
        super(self.__class__, self).__init__(wb_config, wb_defaults=wb_defaults, login_key=login_key, device=device, storage_dir=storage_dir)

    def run_sweep(self, iterations, save_all = False):
      self.save_all = save_all
      sweep_config = self.wb_config['sweep_config']

      # Itialize sweep on wandb server
      sweep_id = wandb.sweep(sweep_config, entity=sweep_config['entity'], project=sweep_config['project'])
        
      # start running on our side, update results to wandb along the way
      wandb.agent(sweep_id, self.train_with_parameters, count=iterations)


    def train_with_parameters(self):
        # Initialize a new wandb run
        with wandb.init(group=self.wb_config['group'], job_type=self.wb_config['job_type']) as run:

            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            run_parameters = wandb.config
            run.name = self.wb_config['sweep_config']['name'] + '-' + run.id
            data = self.get_data(self.wb_config['data'], run)

            input_shape = data['train'].sample_shape()
            class_enums = na.codes_to_enums(data['train'].code_lookup)

            model = self.get_model(self.wb_config['model']['id'], input_shape, class_enums, run_parameters)
            
            optimizer = self.get_optimizer(run_parameters, model)

            model = self.train(model, data['train'], optimizer, run_parameters, run)

            # log validation metric results to wandb
            model.eval()
            with torch.no_grad():
                self.log_metrics(model, 'val', data['val'], run, run_parameters, run_parameters.epochs)
            if self.save_all:
                self.save_model(model, self.wb_config['model'], run_parameters._asdict(), run)

    def train(self, model, train_data, optimizer, config, run: wandb.run):
        train_loader = train_data.get_dataloader(config.batch_size, shuffle=True)
        for epoch in tqdm(range(config.epochs)):
            model.train()
            for batch_idx, (X, y) in enumerate(tqdm(train_loader, leave=False)):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                scores = model(X)
                loss = model.loss(scores, y)
                loss.backward()
                optimizer.step()

        return model


class TrainExperiment(WBExperiment):
    def __init__(self, wb_config, wb_defaults='./.wb.config', login_key=None, device=None, storage_dir='./.wandb_store'):
        super(self.__class__, self).__init__(wb_config, wb_defaults=wb_defaults, login_key=login_key, device=device, storage_dir=storage_dir)
        
    def run_train(self, run_config, save_model=False):
        """Run a training experiment with results logged to weights and biases.
        Useful wb_config keys:
            project: name of the project
            entity: where to send the run. must be jakeval-colab
            config: the config
            save_code: whether to save the mainfile/notebook to W&B
            group: put different runs into one group
            job_type: (train, eval, test, etc)
            name: the run name. can auto-generate
            notes: like a -m in commit

        run_config keys:
            data
                train
                    name
                    version?
                val
                    name
                    version?
            model
                name
            optimizer
                name
                params
                    lr
                    momentum
            train
                epochs
                batch_size
                batch_log_interval
        """
        wb_config_ = {
            'job_type': 'train'
        }
        wb_config_.update(self.wb_config)
        wb_config_['config'] = run_config
        model = None
        with wandb.init(**wb_config_) as run:
            config = run.config
            data = self.get_data(config.data, run)
            run_parameters = _as_named_tuple(config.parameters)
            
            # need wb_config_['config']['model']['id'] otherwise wandb turns 
            # ModelType into a string and isn't comparable to an enum
            model = self.get_model(wb_config_['config']['model']['id'], data['train'].sample_shape(), na.codes_to_enums(data['train'].code_lookup), run_parameters)

            optimizer = self.get_optimizer(run_parameters, model)

            self.train(model, data['train'], data['val'], optimizer, config, run_parameters, run)
            if save_model:
                self.save_model(model, config.model, run_parameters._asdict(), run)
        return model

    def train(self, model, train_data, val_data, optimizer, config, run_parameters, run: wandb.run):
        # Adapted from https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb#scrollTo=j0PV7vvKETBl
        train_loader = train_data.get_dataloader(run_parameters.batch_size, shuffle=True)
        val_loader = val_data.get_dataloader(run_parameters.batch_size, shuffle=False)
        example_count = 0
        accumulate, finalize = self.metric_accumulator() # accumulate metrics as we train
        for epoch in tqdm(range(run_parameters.epochs)):
            model.train()
            for batch_idx, (X, y) in enumerate(tqdm(train_loader, leave=False)):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                scores = model(X)
                ypred = torch.argmax(scores, axis=1)
                accumulate(y, ypred)

                loss = model.loss(scores, y)
                loss.backward()
                optimizer.step()

                example_count += len(X)

                if batch_idx % config['logging']['batch_log_interval'] == 0:
                    # print batch loss and accuracy
                    metrics = finalize()
                    accumulate, finalize = self.metric_accumulator() # reset the accumulator
                    metrics['loss'] = loss
                    print(f'Batch {batch_idx}: Loss: {loss.item():.4f}, Accuracy: {metrics["accuracy"]:.4f}')
                    self.log_progress('train', metrics, example_count, run)

            if config['logging']['eval_every_epoch']:
                # evaluate the model on the validation set at each epoch
                val_metrics = self.validate_model(model, val_loader)
                self.log_progress('val', val_metrics, example_count, run)

    def log_progress(self, split, metrics, example_count, run):
        log_metrics = {}
        for k, v in metrics.items():
            log_metrics[f'{split}/{k}'] = float(v)
        run.log(log_metrics, step=example_count)


class EvalExperiment(WBExperiment):
    def __init__(self, wb_config, wb_defaults='./.wb.config', login_key=None, device=None, storage_dir='./.wandb_store'):
        super(self.__class__, self).__init__(wb_config, wb_defaults=wb_defaults, login_key=login_key, device=device, storage_dir=storage_dir)
        self.device = 'cpu'

    def run_evaluate(self, run_config):
        """
        run_config:
            data
                split (probably val)
                    name
                    version?
            model_name: the-model-name
            eval
                examples
                    k
        """
        wb_config_ = {
            'job_type': 'evaluate'
        }
        wb_config_.update(self.wb_config)
        wb_config_['config'] = run_config
        metrics, examples_df = None, None

        with wandb.init(**wb_config_) as run:
            config = run.config
            data = self.get_data(config.data, run)
            sample_shape = list(data.values())[0].sample_shape()
            model = self.load_model(config.model_name, run, sample_shape, model_version=config.get('model_version', 'latest'))
            data = self.update_data(data, model.class_names)
            metrics, examples_df, model_stats = self.evaluate(model, data, config.get('eval', {}))

            self.log_evaluation(metrics, examples_df, model_stats, run)

        return metrics, examples_df, model_stats

    def evaluate(self, model, data_dict, config):
        metrics = {}
        examples_df = {}
        model_stats = {'weight_norm': self.get_weight_norms(model)}
        for split, data in data_dict.items():
            data_loader = data.get_dataloader(64, shuffle=False)
            metrics[split] = self.validate_model(model, data_loader)
            examples = self.get_hardest_k_examples(model, data, **config.get('examples', {}))
            examples_df[split] = self.get_examples_dataframe(examples, split, data)
        
        return metrics, examples_df, model_stats

    def get_weight_norms(self, model):
        conv_count = 0
        fc_count = 0
        labels = []
        norms = []
        for layer in model.max_norm_layers:
            if isinstance(layer, nn.Conv2d):
                norms.append(layer.weight.norm(2).item())
                labels.append(f"conv{conv_count}")
                conv_count += 1
            if isinstance(layer, nn.Linear):
                norms.append(layer.weight.norm(2).item())
                labels.append(f"linear{fc_count}")
                fc_count += 1
        return [[label, val] for (label, val) in zip(labels, norms)]

    def get_hardest_k_examples(self, model, data, k=32):
        model.eval()
        loader = data.get_dataloader(64, shuffle=False, include_ids=True, include_instrument=True)
        ids = {}
        losses = {}
        predictions = {}
        with torch.no_grad():
            for X, y, id, instrument in loader:
                X, y = X.to(self.device), y.to(self.device)
                scores = model(X)
                loss = model.loss(scores, y, reduction='none').squeeze()
                pred = scores.argmax(dim=1, keepdim=True)
                id = id.squeeze()
                instrument = instrument.squeeze()
                for i in range(loss.shape[0]):
                    inst = instrument[i].item()
                    l = loss[i].item()
                    p = pred[i].item()
                    id_ = id[i].item()
                    if losses.get(inst, 0) < l:
                        losses[inst] = l
                        ids[inst] = id_
                        predictions[inst] = p

        keys = losses.keys()
        losses = np.array([losses[k] for k in keys])
        ids = np.array([ids[k] for k in keys])
        predictions = np.array([predictions[k] for k in keys])

        sorted_idx = np.argsort(losses, axis=0)
        highest_k_losses = losses[sorted_idx[-k:]]
        k_predictions = model.ordinal_to_class_enum(predictions[sorted_idx[-k:]])
        k_ids = ids[sorted_idx[-k:]]
        return {
            'losses': highest_k_losses,
            'predictions': k_predictions,
            'ids': k_ids
        }

    def get_examples_dataframe(self, examples, audio_split, data: na.NsynthDataset):
        df = data.get_dataframe(examples['ids'], audio_split)
        prediction_str = [f"{pred.name} ({pred.value})" for pred in examples['predictions']]
        label = df['family'].to_numpy()
        label_str = [f"{na.InstrumentFamily(code).name} ({na.InstrumentFamily(code).value})" for code in label]

        df['family_pred'] = prediction_str
        df['family_true'] = label_str
        df['loss'] = examples['losses']
        plots = [data.plot_spectrogram(s) for s in df['spectrogram']]
        df['spectrogram'] = [wandb.Image(p, caption=cap) for p, cap in zip(plots, label_str)]
        df['audio'] = ([wandb.Audio(a, sample_rate=sr, caption=cap)
                        for a, sr, cap
                        in zip(df['audio'], df['sample_rate'], label_str)])

        ordered_cols = ['spectrogram', 'audio', 'family_true', 'family_pred', 'loss', 'instrument', 'id']
        return df[ordered_cols]

    def log_evaluation(self, metrics, examples_df, model_stats, run):
        #run.summary.update(metrics)
        for split, split_metrics in metrics.items():
            for metric, value in split_metrics.items():
                run.summary.update({f'{split}/{metric}': value})
        print("logged the metrics")
        for split, df in examples_df.items():
            table = wandb.Table(dataframe=df)
            run.log({
                f'{split}/high-loss-examples': table
            })
        print("logged the examples")

        norm_table = wandb.Table(data=model_stats['weight_norm'], columns=['layer', 'norm'])
        run.log({"weight_norms": wandb.plot.bar(norm_table, 'layer', 'norm', title='Norm of Model Weights')})
        print("logged the weights")
