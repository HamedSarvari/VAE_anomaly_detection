import argparse
import torch
import yaml
from ignite.engine import Engine, Events
from ignite.metrics import Average, RunningAverage
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from VAE import VAEAnomaly
from dataset import load_dataset, get_data_label
from Autoencoder_utils_torch import *
import pandas as pd
import numpy as np
import os

def get_folder_run() -> Path:
    run_path: Path = Path(__file__).parent / 'run'
    if not run_path.exists(): run_path.mkdir()
    i = 0
    while (run_path / str(i)).exists():
        i += 1
    folder_run = run_path / str(i)
    folder_run.mkdir()
    return folder_run


class TrainStep:

    def __init__(self, model, opt, device=None):
        self.model = model
        self.opt = opt
        self.device = device

    def __call__(self, engine, batch):
        x = batch[0]
        if self.device: x.to(self.device)
        pred_output = self.model(x)
        pred_output['loss'].backward()
        self.opt.step()
        return pred_output


def train(model, opt, dloader, epochs: int, experiment_folder, device, args):
    step = TrainStep(model, opt, device)
    trainer = Engine(step)

    Average(lambda o: o['loss']).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda o: o['loss']).attach(trainer, 'running_avg_loss')

    if not args.no_progress_bar: 
        ProgressBar().attach(trainer, ['running_avg_loss'])

    setup_logger(experiment_folder, trainer, model, 
                 args.steps_log_loss, args.steps_log_norm_params)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda e: torch.save(model.state_dict(), experiment_folder / 'model.pth'))

    trainer.run(dloader, epochs)


def setup_logger(experiment_folder, trainer, model, freq_loss: int = 1_000, freq_norm_params: int = 1_000):
    logger = SummaryWriter(log_dir=experiment_folder)
    for l in ['loss', 'kl', 'recon_loss']:
        event_handler = lambda e, l=l: logger.add_scalar(f'train/{l}', e.state.output[l], e.state.iteration)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=freq_loss), event_handler)

    def log_norm(engine, logger, model=model):
        norm1 = sum(p.norm(1) for p in model.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in model.parameters() if p.grad is not None)
        it = engine.state.iteration
        logger.add_scalar('train/norm1_params', norm1, it)
        logger.add_scalar('train/norm1_grad', norm1_grad, it)

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=freq_norm_params), log_norm, logger=logger)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', '-i', type=int, required=True, dest='input_size')
    parser.add_argument('--latent-size', '-l', type=int, required=True, dest='latent_size')
    parser.add_argument('--num-resamples', '-L', type=int, dest='num_resamples', default=10,
                        help='Number of resamples in the latent distribution during training')
    parser.add_argument('--epochs', '-e', type=int, dest='epochs', default=100)
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=32)
    parser.add_argument('--device', '-d', type=str, dest='device', default='cuda:0')
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
    parser.add_argument('--no-progress-bar', action='store_true', dest='no_progress_bar')
    parser.add_argument('--steps-log-loss', type=int, dest='steps_log_loss', default=1_000)
    parser.add_argument('--steps-log-norm-params', type=int, 
                        dest='steps_log_norm_params', default=1_000)
    # Added
    parser.add_argument('--is-mnist', type=bool, dest='is_mnist', default=False)
    parser.add_argument('--datasets', type=str, dest= 'ds_file', default= 'ds_names.csv',
                        help = 'name of the file containing dataset names. one line separated by comma')
    parser.add_argument('--mode', type=str, dest='mode', default = 'unsupervised',
                        choices= ['unsupervised', 'semisupervised'])
    parser.add_argument('--repeats', type=int, dest='num_exps', default=10)
    parser.add_argument('--structure', type=str, dest='net_structure', default='BAE', choices=['An_paper', 'BAE'])

    return parser.parse_args()


def store_codebase_into_experiment(experiment_folder):
    with open(Path(__file__).parent / 'VAE.py') as f:
        code = f.read()
    with open(experiment_folder / 'vae.py', 'w') as f:
        f.write(code)

if __name__ == '__main__':

    args = get_args()
    print(args)
    mode = args.mode
    is_mnist = args.is_mnist
    print('Running in ' + mode + ' setting')
    ds_names = pd.read_csv(args.ds_file)
    print(list(ds_names))
    num_exps = args.num_exps


    for ds_name in ds_names:

        print('Dataset name: ------------------', ds_name ,'-----------------------------------------------')

        experiment_folder = get_folder_run()
        print(experiment_folder)


        if mode == 'unsupervised':
            data, data_t, labels = get_data_label(ds_name, mnist=args.is_mnist)
            dloader = DataLoader(load_dataset(ds_name, inliers=False, mnist= args.is_mnist), args.batch_size)

        elif mode == 'semisupervised':
            data, data_t, labels = get_data_label(ds_name, mnist= args.is_mnist, inliers=True)
            dloader = DataLoader(load_dataset(ds_name, inliers=True, mnist=args.is_mnist), args.batch_size)
        else:
            break

        model = VAEAnomaly(data.shape[1], args.latent_size, args.net_structure, args.num_resamples).to(args.device)
        opt = torch.optim.Adam(model.parameters(), args.lr)

        store_codebase_into_experiment(experiment_folder)
        with open(experiment_folder / 'config.yaml', 'w') as f:
            yaml.dump(args, f)

        results = []
        for i in range(num_exps):  # Number of experiment reps

            train(model, opt, dloader, args.epochs, experiment_folder, args.device, args)

            # evaluation phase:
            # load all data, regardless of mode in training to evaluate
            data, data_t, labels = get_data_label(ds_name, mnist=is_mnist)
            rec_probs = model.reconstructed_probability(data_t)

            rec_probs_pd = pd.DataFrame(rec_probs)

            folder_path = experiment_folder + '/' + ds_name + '/'
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            rec_probs_pd.to_csv(folder_path + ds_name +'_' + mode + '_rep' + str(i) + '.csv')


            AUCPR = eval_model(data_to_dic(np.array(-rec_probs)), data_to_dic(labels))  #higher prob = less outlierness
            print('AUCPR= ', AUCPR)
            results.append(AUCPR)

        results.append('---')
        tmp = results[:-1]
        results.append(np.mean(tmp))
        results.append(np.std(tmp))
        df = pd.DataFrame(results)
        df.to_csv(folder_path + ds_name + '_' + mode + '_AUCPRs.csv', header=False, index=False)


