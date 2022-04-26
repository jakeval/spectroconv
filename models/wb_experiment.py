import torch
from models import cnn_model
import wandb
from tqdm.auto import tqdm
from data_utils import nsynth_adapter as na
from models import cnn_model
from models import lrlc_model_debug
import json
import os
import pickle
import enum


class ModelType(enum.IntEnum):
    CNN = 0
    LRLC = 1


class WBExperiment:
    def __init__(self, wb_config, wb_defaults='./.wb.config', wb_key=None, device=None, storage_dir='./.wandb_store'):
        if wb_key is not None:
            wandb.login(key=None)
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

    def validate_model(self, model, data_loader):
        model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                scores = model(X)
                loss += model.loss(scores, y, reduction='sum')  # sum up batch loss
                pred = scores.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum()

        loss /= len(data_loader.dataset)
        accuracy = correct / len(data_loader.dataset)
        return {
            'loss': loss,
            'accuracy': accuracy
        }

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

    def get_model(self, config, class_enums, input_shape):
        if config['type_code'] == ModelType.CNN:
            return cnn_model.CnnClf(input_shape, class_enums).float().to(self.device)
        if config['type_code'] == ModelType.LRLC:
            return lrlc_model_debug.LrlcClfDebug(config['rank'], input_shape, class_enums, local_dim=config.get('local_dim', None)).float().to(self.device)

    def load_model(self, model_name, run, input_shape, model_version='latest'):
        model_artifact = run.use_artifact(f"{model_name}:{model_version}")
        model_config = model_artifact.metadata
        model_dir = model_artifact.download(root=self.storage_dir)
        model_path = os.path.join(model_dir, model_config['model_path'])
        class_names_path = os.path.join(model_dir, model_config['class_names_path'])
        code_lookup = None
        with open(class_names_path, 'rb') as f:
            code_lookup = pickle.load(f)
        model = self.get_model(model_config, code_lookup, input_shape)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        return model

    def save_model(self, model, model_config, run):
        model_name = model_config["name"]
        model_filename = f"{model_name}.pth"
        class_names_filename = f"{model_name}.pkl"
        model_file = os.path.join(self.storage_dir, model_filename)
        class_names_file = os.path.join(self.storage_dir, class_names_filename)
        metadata = dict(model_config)
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


class TrainExperiment(WBExperiment):
    def __init__(self, wb_config, wb_defaults='./.wb.config', wb_key=None, device=None, storage_dir='./.wandb_store'):
        super(self.__class__, self).__init__(wb_config, wb_defaults=wb_defaults, wb_key=wb_key, device=device, storage_dir=storage_dir)

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
            model = self.get_model(config.model, na.codes_to_enums(data['train'].code_lookup), data['train'].sample_shape())
            optimizer = self.get_optimizer(config.optimizer, model)
            self.train(model, data['train'], data['val'], optimizer, config.train, run)
            if save_model:
                self.save_model(model, config.model, run)
        return model

    def train(self, model, train_data, val_data, optimizer, config, run: wandb.run):
        # Adapted from https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb#scrollTo=j0PV7vvKETBl
        train_loader = train_data.get_dataloader(config['batch_size'], shuffle=True)
        val_loader = val_data.get_dataloader(config['batch_size'], shuffle=False)
        example_count = 0
        train_accuracy = 0
        train_accuracy_samples = 0
        for epoch in tqdm(range(config['epochs'])):
            model.train()
            for batch_idx, (X, y) in enumerate(tqdm(train_loader, leave=False)):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                scores = model(X)
                ypred = torch.argmax(scores, axis=1)
                train_accuracy += (ypred == y).sum().item()
                loss = model.loss(scores, y)
                loss.backward()
                optimizer.step()

                example_count += len(X)
                train_accuracy_samples += len(X)

                if batch_idx % config['batch_log_interval'] == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)\tLoss: {loss.item():.6f}')
                    self.log_progress("train", loss, train_accuracy / train_accuracy_samples, example_count, epoch, run)
                    train_accuracy = 0
                    train_accuracy_samples = 0

            # evaluate the model on the validation set at each epoch
            metrics = self.validate_model(model, val_loader)
            self.log_progress("validation", metrics['loss'], metrics['accuracy'], example_count, epoch, run)

    def log_progress(self, split, loss, accuracy, example_count, epoch, run):
        loss = float(loss)
        accuracy = float(accuracy)
        run.log({"epoch": epoch, f"{split}/loss": loss, f"{split}/accuracy": accuracy}, step=example_count)

    def get_optimizer(self, config, model):
        return getattr(torch.optim, config['name'])(model.parameters(), **config['params'])


class EvalExperiment(WBExperiment):
    def __init__(self, wb_config, wb_defaults='./.wb.config', wb_key=None, device=None, storage_dir='./.wandb_store'):
        super(self.__class__, self).__init__(wb_config, wb_defaults=wb_defaults, wb_key=wb_key, device=device, storage_dir=storage_dir)

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
            metrics, examples_df = self.evaluate(model, data, config.get('eval', {}))
            self.log_evaluation(metrics, examples_df, run)
        return metrics, examples_df

    def evaluate(self, model, data_dict, config):
        metrics = {}
        examples_df = {}
        for split, data in data_dict.items():
            data_loader = data.get_dataloader(64, shuffle=False)
            metrics[split] = self.validate_model(model, data_loader)
            examples = self.get_hardest_k_examples(model, data, **config.get('examples', {}))
            examples_df[split] = self.get_examples_dataframe(examples, split, data)
        return metrics, examples_df

    def get_hardest_k_examples(self, model, data, k=32):
        model.eval()
        loader = data.get_dataloader(64, shuffle=False, include_ids=True)
        losses = None
        predictions = None
        ids = None
        with torch.no_grad():
            for X, y, id in loader:
                X, y = X.to(self.device), y.to(self.device)
                scores = model(X)
                loss = model.loss(scores, y, reduction='none').view((-1, 1))
                pred = scores.argmax(dim=1, keepdim=True)

                if losses is None:
                    losses = loss
                    predictions = pred
                    ids = id
                else:
                    losses = torch.cat((losses, loss), 0)
                    predictions = torch.cat((predictions, pred), 0)
                    ids = torch.cat((ids, id), 0)

        sorted_idx = torch.argsort(losses, dim=0).squeeze()
        highest_k_losses = losses[sorted_idx[-k:]]
        k_predictions = model.ordinal_to_class_enum(predictions[sorted_idx[-k:]])
        k_ids = ids[sorted_idx[-k:]]
        return {
            'losses': highest_k_losses,
            'predictions': k_predictions,
            'ids': k_ids.squeeze().numpy()
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

    def log_evaluation(self, metrics, examples_df, run):
        run.summary.update(metrics)
        for split, df in examples_df.items():
            table = wandb.Table(dataframe=df)
            run.log({
                f'{split}/high-loss-examples': table
            })
