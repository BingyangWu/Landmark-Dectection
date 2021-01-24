import time
import datetime
import os
import pandas as pd
import numpy as np
import json

import torch
import torch.utils.data as td
import torch.nn.modules.distance
import torch.optim as optim
import torch.nn.functional as F

import config as cfg
from datasets import rhpe, rsna
from constants import TRAIN, VAL
from utils import log, io, nn
from utils.nn import to_numpy, Batch
from train_aae_unsupervised import AAETraining
from landmarks import lmutils, lmvis, fabrec
import landmarks.lmconfig as lmcfg
import aae_training


class AAELandmarkTraining(AAETraining):
    def __init__(self, datasets, args, session_name="debug", **kwargs):
        args.reset = (
            False  # just to make sure we don't reset the discriminator by accident
        )
        try:
            ds = datasets[TRAIN]
        except KeyError:
            ds = datasets[VAL]
        self.num_landmarks = ds.NUM_LANDMARKS
        self.all_landmarks = ds.ALL_LANDMARKS
        self.skeleton = ds.SKELETON

        super().__init__(datasets, args, session_name, macro_batch_size=0, **kwargs)

        self.optimizer_lm_head = optim.Adam(
            self.saae.LMH.parameters(), lr=args.lr_heatmaps, betas=(0.9, 0.999)
        )
        self.optimizer_E = optim.Adam(
            self.saae.Q.parameters(), lr=0.00002, betas=(0.9, 0.999)
        )
        # self.optimizer_G = optim.Adam(self.saae.P.parameters(), lr=0.00002, betas=(0.9, 0.999))

    def _get_network(self, pretrained):
        return fabrec.Fabrec(
            self.num_landmarks,
            input_size=self.args.input_size,
            z_dim=self.args.embedding_dims,
        )

    @staticmethod
    def print_eval_metrics(nmes, show=False):
        def ced_curve(_nmes):
            y = []
            x = np.linspace(0, 10, 50)
            for th in x:
                recall = 1.0 - lmutils.calc_landmark_failure_rate(_nmes, th)
                recall *= 1 / len(x)
                y.append(recall)
            return x, y

        def auc(recalls):
            return np.sum(recalls)

        # for err_scale in np.linspace(0.1, 1, 10):
        for err_scale in [1.0]:
            # print('\nerr_scale', err_scale)
            # print(np.clip(lm_errs_max_all, a_min=0, a_max=10).mean())

            fr = lmutils.calc_landmark_failure_rate(nmes * err_scale)
            X, Y = ced_curve(nmes)

            log.info("NME:   {:>6.3f}".format(nmes.mean() * err_scale))
            log.info(
                "FR@10: {:>6.3f} ({})".format(fr * 100, np.sum(nmes.mean(axis=1) > 10))
            )
            log.info("AUC:   {:>6.4f}".format(auc(Y)))
            # log.info('NME:   {nme:>6.3f}, FR@10: {fr:>6.3f} ({fc}), AUC:   {auc:>6.4f}'.format(
            #     nme=nmes.mean()*err_scale,
            #     fr=fr*100,
            #     fc=np.sum(nmes.mean(axis=1) > 10),
            #     auc=auc(Y)))

            if show:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2)
                axes[0].plot(X, Y)
                axes[1].hist(nmes.mean(axis=1), bins=20)
                plt.show()

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]
        nmes = current.get("nmes", np.zeros(0))

        str_stats = [
            "[{ep}][{i}/{iters_per_epoch}] "
            "l_rec={avg_loss_recon:.3f} "
            # 'ssim={avg_ssim:.3f} '
            # 'ssim_torch={avg_ssim_torch:.3f} '
            # 'z_mu={avg_z_recon_mean: .3f} '
            "l_lms={avg_loss_lms:.4f} "
            "err_lms={avg_err_lms_all:.2f} "
            "{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})"
        ][0]
        log.info(
            str_stats.format(
                ep=current["epoch"] + 1,
                i=current["iter"] + 1,
                iters_per_epoch=self.iters_per_epoch,
                avg_loss=means.get("loss", -1),
                avg_loss_recon=means.get("loss_recon", -1),
                avg_ssim=1.0 - means.get("ssim", -1),
                avg_ssim_torch=means.get("ssim_torch", -1),
                avg_loss_activations=means.get("loss_activations", -1),
                avg_loss_lms=means.get("loss_lms", -1),
                avg_z_l1=means.get("z_l1", -1),
                avg_z_recon_mean=means.get("z_recon_mean", -1),
                t=means["iter_time"],
                t_data=means["time_dataloading"],
                t_proc=means["time_processing"],
                avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
                total_iter=self.total_iter + 1,
                total_time=str(datetime.timedelta(seconds=self._training_time())),
            )
        )

    def _print_epoch_summary(
        self, epoch_stats, epoch_starttime, batch_samples, eval=False
    ):
        means = pd.DataFrame(epoch_stats).mean().to_dict()

        try:
            nmes = np.concatenate([s["nmes"] for s in self.epoch_stats if "nmes" in s])
        except (KeyError, ValueError):
            nmes = np.zeros((1, 100))

        duration = int(time.time() - epoch_starttime)
        log.info("{}".format("-" * 100))
        str_stats = [
            "           "
            "l_rec={avg_loss_recon:.3f} "
            # 'ssim={avg_ssim:.3f} '
            # 'ssim_torch={avg_ssim_torch:.3f} '
            # 'z_mu={avg_z_recon_mean:.3f} '
            "l_lms={avg_loss_lms:.4f} "
            "err_lms={avg_err_lms_all:.2f} "
            "\tT: {time_epoch}"
        ][0]
        log.info(
            str_stats.format(
                iters_per_epoch=self.iters_per_epoch,
                avg_loss=means.get("loss", -1),
                avg_loss_recon=means.get("loss_recon", -1),
                avg_ssim=1.0 - means.get("ssim", -1),
                avg_ssim_torch=means.get("ssim_torch", -1),
                avg_loss_lms=means.get("loss_lms", -1),
                avg_loss_lms_cnn=means.get("loss_lms_cnn", -1),
                avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
                avg_z_recon_mean=means.get("z_recon_mean", -1),
                t=means["iter_time"],
                t_data=means["time_dataloading"],
                t_proc=means["time_processing"],
                total_iter=self.total_iter + 1,
                total_time=str(datetime.timedelta(seconds=self._training_time())),
                time_epoch=str(datetime.timedelta(seconds=duration)),
            )
        )
        try:
            recon_errors = np.concatenate(
                [stats["l1_recon_errors"] for stats in self.epoch_stats]
            )
            rmse = np.sqrt(np.mean(recon_errors ** 2))
            log.info("RMSE: {} ".format(rmse))
        except KeyError:
            # print("no l1_recon_error")
            pass

        if self.args.eval and nmes is not None:
            # benchmark_mode = hasattr(self.args, 'benchmark')
            # self.print_eval_metrics(nmes, show=benchmark_mode)
            self.print_eval_metrics(nmes, show=False)

        # Saving output images
        batch = batch_samples["batch"]
        X_recon = batch_samples["X_recon"]
        X_lm_hm = batch_samples["X_lm_hm"]
        lm_preds_max = batch_samples["lm_preds_max"]
        ds = batch_samples["ds"]

        out_dir = os.path.join(
            cfg.REPORT_DIR,
            "landmark_predictions",
            self.session_name,
            str(self.epoch + 1),
            "eval" if eval else "train",
        )
        io.makedirs(out_dir)
        lmvis.visualize_batch(
            batch.images,
            batch.landmarks,
            X_recon,
            X_lm_hm,
            lm_preds_max,
            self.all_landmarks,
            lm_heatmaps=batch.lm_heatmaps,
            target_images=batch.target_images,
            ds=ds,
            ocular_norm=self.args.ocular_norm,
            clean=False,
            overlay_heatmaps_input=False,
            overlay_heatmaps_recon=False,
            f=1.0,
            wait=self.wait,
            skeleton=self.skeleton,
            out_dir=out_dir,
        )

        # Saving loss
        out_dir = os.path.join(cfg.REPORT_DIR, "losses", self.session_name)
        filename = "loss_eval.json" if eval else "loss_train.json"
        try:
            with open(os.path.join(out_dir, filename), "r") as outfile:
                data = json.load(outfile)
        except (json.decoder.JSONDecodeError, FileNotFoundError):
            data = dict()
        losses = {}
        losses.update(
            avg_loss=means.get("loss", -1),
            avg_loss_recon=means.get("loss_recon", -1),
            avg_ssim=1.0 - means.get("ssim", -1),
            avg_ssim_torch=means.get("ssim_torch", -1),
            avg_loss_lms=means.get("loss_lms", -1),
            avg_loss_lms_cnn=means.get("loss_lms_cnn", -1),
            avg_err_lms_all=nmes[:, self.all_landmarks].mean(),
            avg_z_recon_mean=means.get("z_recon_mean", -1),
        )
        data[str(self.epoch + 1)] = losses
        io.makedirs(out_dir)
        with open(os.path.join(out_dir, filename), "w") as outfile:
            json.dump(data, outfile)

    def eval_epoch(self):
        log.info("")
        log.info("Evaluating '{}'...".format(self.session_name))
        # log.info("")

        self.epoch_stats = []
        self.saae.eval()

        self._run_epoch(self.datasets[VAL], eval=True)
        # print average loss and accuracy over epoch
        return self.epoch_stats

    def train(self, num_epochs=None):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        # log.info("")

        while num_epochs is None or self.epoch < num_epochs:
            log.info("")
            log.info("Epoch {}/{}".format(self.epoch + 1, num_epochs))
            log.info("=" * 10)

            self.epoch_stats = []

            self._run_epoch(self.datasets[TRAIN])

            # save model every few epochs
            if (self.epoch + 1) % self.snapshot_interval == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch

            if self._is_eval_epoch():
                self.eval_epoch()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        log.info(
            "Training completed in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def _run_epoch(self, dataset, eval=False):
        epoch_starttime = time.time()
        batchsize = self.args.batchsize_eval if eval else self.args.batchsize
        self.iters_per_epoch = int(len(dataset) / batchsize)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0
        dataloader = td.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=not eval,
            num_workers=self.workers,
            drop_last=not eval,
        )
        batch_samples = None
        for data in dataloader:
            batch = Batch(data, eval=eval)
            batch_samples = self._run_batch(batch, eval=eval)
            self.total_iter += 1
            self.saae.total_iter = self.total_iter
            self.iter_in_epoch += 1

        self._print_epoch_summary(
            self.epoch_stats, epoch_starttime, eval=eval, batch_samples=batch_samples
        )

    def _run_batch(self, batch, eval=False, ds=None):
        time_dataloading = time.time() - self.iter_starttime
        time_proc_start = time.time()
        iter_stats = {"time_dataloading": time_dataloading}

        self.saae.zero_grad()
        self.saae.eval()

        input_images = (
            batch.target_images if batch.target_images is not None else batch.images
        )

        with torch.set_grad_enabled(self.args.train_encoder):
            z_sample = self.saae.Q(input_images)

        iter_stats.update({"z_recon_mean": z_sample.mean().item()})

        #######################
        # Reconstruction phase
        #######################
        with torch.set_grad_enabled(self.args.train_encoder and not eval):
            X_recon = self.saae.P(z_sample)

        # calculate reconstruction error for debugging and reporting
        with torch.no_grad():
            iter_stats["loss_recon"] = aae_training.loss_recon(batch.images, X_recon)

        #######################
        # Landmark predictions
        #######################
        train_lmhead = not eval
        lm_preds_max = None
        with torch.set_grad_enabled(train_lmhead):
            self.saae.LMH.train(train_lmhead)
            X_lm_hm = self.saae.LMH(self.saae.P)
            if batch.lm_heatmaps is not None:
                loss_lms = F.mse_loss(batch.lm_heatmaps, X_lm_hm) * 100 * 3
                iter_stats.update({"loss_lms": loss_lms.item()})

            if eval or self._is_printout_iter(eval):
                # expensive, so only calculate when every N iterations
                # X_lm_hm = lmutils.decode_heatmap_blob(X_lm_hm)
                X_lm_hm = lmutils.smooth_heatmaps(X_lm_hm)
                lm_preds_max = self.saae.heatmaps_to_landmarks(X_lm_hm)

            if eval or self._is_printout_iter(eval):
                lm_gt = to_numpy(batch.landmarks)
                nmes = lmutils.calc_landmark_nme(
                    lm_gt,
                    lm_preds_max,
                    ocular_norm=self.args.ocular_norm,
                    image_size=self.args.input_size,
                )
                # nccs = lmutils.calc_landmark_ncc(batch.images, X_recon, lm_gt)
                iter_stats.update({"nmes": nmes})

        if train_lmhead:
            # if self.args.train_encoder:
            #     loss_lms = loss_lms * 80.0
            loss_lms.backward()
            self.optimizer_lm_head.step()
            if self.args.train_encoder:
                self.optimizer_E.step()
                # self.optimizer_G.step()

        # statistics
        iter_stats.update(
            {
                "epoch": self.epoch,
                "timestamp": time.time(),
                "iter_time": time.time() - self.iter_starttime,
                "time_processing": time.time() - time_proc_start,
                "iter": self.iter_in_epoch,
                "total_iter": self.total_iter,
                "batch_size": len(batch),
            }
        )
        self.iter_starttime = time.time()

        self.epoch_stats.append(iter_stats)

        batch_samples = {
            "batch": batch,
            "X_recon": X_recon,
            "X_lm_hm": X_lm_hm,
            "lm_preds_max": lm_preds_max,
            "ds": ds,
        }

        # print stats every N mini-batches
        if self._is_printout_iter(eval):
            self._print_iter_stats(self.epoch_stats[-self._print_interval(eval) :])

            out_dir = os.path.join(
                cfg.REPORT_DIR,
                "landmark_predictions",
                self.session_name,
                str(self.epoch + 1),
            )
            io.makedirs(out_dir)

            lmvis.visualize_batch(
                batch.images,
                batch.landmarks,
                X_recon,
                X_lm_hm,
                lm_preds_max,
                self.all_landmarks,
                lm_heatmaps=batch.lm_heatmaps,
                target_images=batch.target_images,
                ds=ds,
                ocular_norm=self.args.ocular_norm,
                clean=False,
                overlay_heatmaps_input=False,
                overlay_heatmaps_recon=False,
                f=1.0,
                wait=self.wait,
                skeleton=self.skeleton,
            )
        return batch_samples


def run():

    from utils import random

    if args.seed is not None:
        random.init_random(args.seed)
    # log.info(json.dumps(vars(args), indent=4))

    datasets = {}
    for phase, dsnames, num_samples in zip(
        (TRAIN, VAL),
        (args.dataset_train, args.dataset_val),
        (args.train_count, args.val_count),
    ):
        train = phase == TRAIN
        name = dsnames[0]
        transform = nn.build_transform(deterministic=not train, daug=args.daug)
        root, cache_root = cfg.get_dataset_paths(name)
        dataset_cls = cfg.get_dataset_class(name)
        datasets[phase] = dataset_cls(
            root=root,
            cache_root=cache_root,
            train=train,
            max_samples=num_samples,
            use_cache=args.use_cache,
            start=args.st,
            align_face_orientation=args.align,
            crop_source=args.crop_source,
            return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP,
            with_occlusions=args.occ and train,
            landmark_sigma=args.sigma,
            transform=transform,
            image_size=args.input_size,
        )
        print(datasets[phase])

    fntr = AAELandmarkTraining(
        datasets,
        args,
        session_name=args.sessionname,
        snapshot_interval=args.save_freq,
        workers=args.workers,
        wait=args.wait,
    )

    torch.backends.cudnn.benchmark = True
    if args.eval:
        fntr.eval_epoch()
    else:
        fntr.train(num_epochs=args.epochs)


if __name__ == "__main__":

    import sys
    import configargparse

    np.set_printoptions(linewidth=np.inf)

    # Disable traceback on Ctrl+c
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = configargparse.ArgParser()
    defaults = {"batchsize": 50, "train_encoder": False, "train_decoder": False}
    aae_training.add_arguments(parser, defaults)

    # Dataset
    parser.add_argument(
        "--dataset",
        default=["w300"],
        type=str,
        help="dataset for training and testing",
        choices=["rhpe", "rsna"],
        nargs="+",
    )

    # Landmarks
    parser.add_argument(
        "--lr-heatmaps",
        default=0.001,
        type=float,
        help="learning rate for landmark heatmap outputs",
    )
    parser.add_argument(
        "--sigma", default=7, type=float, help="size of landmarks in heatmap"
    )
    parser.add_argument(
        "-n",
        "--ocular-norm",
        default=lmcfg.LANDMARK_OCULAR_NORM,
        type=str,
        help="how to normalize landmark errors",
        choices=["pupil", "outer", "none"],
    )

    args = parser.parse_args()

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = "debug"

    run()
