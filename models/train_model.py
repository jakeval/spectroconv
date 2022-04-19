import torch
from torch import nn
import numpy as np
from torch import optim
from models import cnn_model
from sklearn.metrics import f1_score
import wandb
from tqdm.auto import tqdm
from data_utils import nsynth_adapter as na
from models import cnn_model
import json
import os
import pickle
from matplotlib import pyplot as plt


class WBExperiment:
    def __init__(self, wb_defaults='./.wb.config', wb_key=None, device=None, storage_dir='./.wandb_store'):
        if wb_key is not None:
            wandb.login(key=None)
        else:
            wandb.login()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if isinstance(wb_defaults, str):
            with open(wb_defaults) as f:
                self.wb_config = json.load(f)
        elif isinstance(wb_defaults, dict):
            self.wb_config = wb_defaults
        else:
            self.wb_config = {}
        self.storage_dir = storage_dir

    def run_train(self, wb_config, run_config, save_model=True):
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
                val
            model: None
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
        wb_config_ = self.wb_config.copy()
        wb_config_['job_type'] = 'train'
        wb_config_.update(wb_config)
        wb_config_['config'] = run_config
        with wandb.init(**wb_config_) as run:
            config = run.config
            data = self.get_data(config.data, run)
            model = self.get_model(config.model, na.codes_to_enums(data['train'].code_lookup), data['train'].sample_shape())
            optimizer = self.get_optimizer(config.optimizer, model)
            self.train(model, data['train'], data['val'], optimizer, config.train, run)
            if save_model:
                self.save_model(model, config.model, run)

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

    def run_evaluate(self, wb_config, run_config):
        wb_config_ = self.wb_config.copy()
        wb_config_['job_type'] = 'evaluate'
        wb_config_.update(wb_config)
        wb_config_['config'] = run_config
        metrics, examples_df = None, None
        with wandb.init(**wb_config_) as run:
            config = run.config
            data = self.get_data(config.data, run)
            sample_shape = list(data.values())[0].sample_shape()
            model = self.load_model(config.model_name, run, sample_shape, model_version=config.get('model_version', 'latest'))
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

    def log_progress(self, split, loss, accuracy, example_count, epoch, run):
        loss = float(loss)
        accuracy = float(accuracy)
        run.log({"epoch": epoch, f"{split}/loss": loss, f"{split}/accuracy": accuracy}, step=example_count)

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

    def get_model(self, config, class_enums, input_shape):
        return cnn_model.CnnClf(input_shape, class_enums).to(self.device)

    def get_optimizer(self, config, model):
        return getattr(torch.optim, config['name'])(model.parameters(), **config['params'])

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

    def log_evaluation(self, metrics, examples_df, run):
        run.summary.update(metrics)
        for split, df in examples_df.items():
            table = wandb.Table(dataframe=df)
            run.log({
                f'{split}/high-loss-examples': table
            })

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


def train_model_dataloader(clf: cnn_model.CnnClf, dataloader, dataloader_val, lr=0.005, momentum=0.9, epochs=80, device = None):
    softmax = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum)
    val_iter = iter(dataloader_val)

    print("Start Training...")
    losses = []
    train_accuracies = []
    val_accuracies = []
    val_f1s = []
    print_increments = int(epochs // 10)
    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if device:
              inputs = inputs.to(device)
              labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = clf(inputs)
            loss = softmax(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            detached_outputs = outputs.detach()
            detached_labels = labels.detach()
            if device:
              detached_outputs = detached_outputs.cpu()
              detached_labels = detached_labels.cpu()
            running_accuracy += (clf.predict_from_scores(detached_outputs.numpy()) == detached_labels.numpy()).mean()
        
        epoch_loss = running_loss / (i+1)
        epoch_accuracy = running_accuracy / (i+1)
        with torch.no_grad():
            train_accuracies.append(epoch_accuracy)
            val_data = next(val_iter, None)
            if val_data is None:
                val_iter = iter(dataloader_val)
                val_data = next(dataloader_val)
            val_X, val_y = val_data

            detached_outputs = outputs.detach()
            detached_labels = labels.detach()
            if device:
              detached_outputs = detached_outputs.cpu()
              detached_labels = detached_labels.cpu()

            running_accuracy += (clf.predict_from_scores(detached_outputs.numpy()) == detached_labels.numpy()).mean()
            
            val_accuracy = stream_accuracy(clf, dataloader_val, device)
            val_accuracies.append(val_accuracy)
            val_f1 = stream_f1(clf, dataloader_val, device)
            val_f1s.append(val_f1)
        
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f} \t val f1: {val_f1s[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies, val_f1s


def train_model(clf: cnn_model.CnnClf, X, y, X_val, y_val, lr=0.005, momentum=0.9, epochs=80):
    X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1], X_val.shape[2]))
    trainloader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=min(64, X.shape[0]), shuffle=True)
    softmax = nn.CrossEntropyLoss()
    optimizer = optim.SGD(clf.parameters(), lr=lr, momentum=momentum)
    
    print("Check starting accuracy...")
    print(get_accuracy(clf, X, y), get_accuracy(clf, X_val, y_val))

    print("Start Training...")
    losses = []
    train_accuracies = []
    val_accuracies = []
    print_increments = int(epochs // 10)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = clf(inputs)
            loss = softmax(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss / (i+1)
        with torch.no_grad():
            train_accuracies.append(get_accuracy(clf, X, y, size=128))
            val_accuracies.append(get_accuracy(clf, X_val, y_val, size=128))
        if print_increments == 0 or (epoch % print_increments == print_increments-1):
            print(f'epoch {epoch + 1} \t\t loss: {epoch_loss:.3f} \t train: {train_accuracies[-1]:.3f} \t val: {val_accuracies[-1]:.3f}')
        losses.append(epoch_loss)

    return clf, losses, train_accuracies, val_accuracies


def get_accuracy(clf: cnn_model.CnnClf, X, y, size=None, device = None):
    if size is not None:
        idx = np.random.choice(np.arange(X.shape[0]), size=size, replace=False)
        return (clf.predict(X[idx], device) == y[idx]).mean()
    else:
        return (clf.predict(X, device) == y).mean()

def get_f1(clf: cnn_model.CnnClf, X, y, size=None, device = None):
  predictions = clf.predict(X, device)
  #print(f1_score(y, predictions, average = None))
  #print(f1_score(y, predictions, average = "macro"))
  return f1_score(y, predictions, average = "macro")



def stream_accuracy(clf, dataloader, device):
    accuracies = []
    batch_sizes = []
    for data in dataloader:
        X, y = data
        accuracies.append(get_accuracy(clf, X.numpy(), y.numpy(), device = device))
        batch_sizes.append(X.shape[0])
    accuracies = np.array(accuracies)
    batch_sizes = np.array(batch_sizes)
    weights = batch_sizes / batch_sizes.sum()
    return accuracies @ weights


def stream_f1(clf, dataloader, device):
    f1s = []
    batch_sizes = []
    for data in dataloader:
        X, y = data
        f1s.append(get_f1(clf, X.numpy(), y.numpy(), device = device))
        batch_sizes.append(X.shape[0])
    f1s = np.array(f1s)
    batch_sizes = np.array(batch_sizes)
    weights = batch_sizes / batch_sizes.sum()
    return f1s @ weights