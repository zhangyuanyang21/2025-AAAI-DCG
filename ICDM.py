from sklearn.utils import shuffle
from loss import *
from evaluation import evaluation
from util import next_batch
import numpy as np
from baseModels import *
from util import target_l2, normalize
import matplotlib.pyplot as plt
import scipy.io
import os
import datetime
import seaborn as sns
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
class icdm():

    def __init__(self, config):

        self._config = config
        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.df1 = Unet(config['diffusion']['emb_size'], config['diffusion']['time_type'], config['diffusion']['out_dim1'])
        self.df2 = Unet(config['diffusion']['emb_size'], config['diffusion']['time_type'], config['diffusion']['out_dim2'])
        self.noise_scheduler = NoiseScheduler(config['noise_scheduler']['num_timesteps'])
        self.clusterLayer = ClusterProject(config['Autoencoder']['arch1'][-1], config['training']['n_clusters'])
        self.AttentionLayer =AttentionLayer(config['Autoencoder']['arch1'][-1])

    def to_device(self, device):
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.df1.to(device)
        self.df2.to(device)
        self.clusterLayer.to(device)
        self.AttentionLayer.to(device)

    def train(self, config, x1_train, x2_train, Y_list, mask, optimizer, device):

        criterion_cluster = ClusterLoss(config['training']['n_clusters'], 0.5, device).to(device)
        flag_1 = (torch.LongTensor([1, 1]).to(device) == mask).int()
        Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0
        accs = []
        nmis = []
        aris = []
        for epoch in range(config['training']['epoch'] + 1):
            X1, X2, X3, X4 = shuffle(x1_train, x2_train, flag_1[:, 0], flag_1[:, 1])
            loss_all, loss_rec1, loss_rec2, loss_mmi, loss_df, loss_cluster, loss_hc, loss_ce = 0, 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, x1_index, x2_index, batch_No in next_batch(X1, X2, X3, X4,
                                                                               config['training']['batch_size']):
                if len(batch_x1) == 1:
                    continue
                index_both = x1_index + x2_index == 2  # C in indicator matrix A of complete multi-view data
                z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])  # [Z_C^1;Z_I^1]
                z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])  # [Z_C^2;Z_I^2]
                if len(batch_x1[index_both])== 1:
                    continue
                z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
                z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_view1_both), batch_x1[index_both])
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_view2_both), batch_x2[index_both])
                rec_loss = (recon1 + recon2)
                criterion_instance = InstanceLoss(z_view1_both.shape[0], 1.0, device).to(device)
                h_both = self.AttentionLayer(z_view1_both, z_view2_both)
                mmi_loss = MMI(h_both, z_view1_both) + MMI(h_both, z_view2_both)
                y1, p1 = self.clusterLayer(z_view1_both)
                y2, p2 = self.clusterLayer(z_view2_both)
                cluster_loss = criterion_cluster(y1, y2)
                y, _ = self.clusterLayer(h_both)
                y_max = torch.maximum(y1, y2)
                y_max = torch.maximum(y_max, y)
                y_max = target_l2(y_max)
                y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
                hc_loss = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
                noise_1 = torch.randn(z_1.shape).to(device)
                timesteps1 = torch.randint(
                    0, config['noise_scheduler']['num_timesteps'], (z_1.shape[0],)
                ).long()
                timesteps1 = timesteps1.to(device)
                noisy_1 = self.noise_scheduler.add_noise(z_1, noise_1, timesteps1, device)
                noise_pred_1 = self.df1(noisy_1, timesteps1)
                dfloss1 = F.mse_loss(noise_pred_1, noise_1).to(device)
                noise_2 = torch.randn(z_2.shape).to(device)
                timesteps2 = torch.randint(
                    0, config['noise_scheduler']['num_timesteps'], (z_2.shape[0],)
                ).long()
                noisy_2 = self.noise_scheduler.add_noise(z_2, noise_2, timesteps2, device)
                noise_pred_2 = self.df2(noisy_2, timesteps2)
                dfloss2 = F.mse_loss(noise_pred_2, noise_2).to(device)
                dfloss = dfloss1 + dfloss2
                v1_missing_latent_eval = z_view1_both
                v2_missing_latent_eval = z_view2_both
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for i, t in enumerate(timesteps):
                    t1 = torch.from_numpy(np.repeat(t, v1_missing_latent_eval.shape[0])).long().to(device)
                    t2 = torch.from_numpy(np.repeat(t, v2_missing_latent_eval.shape[0])).long().to(device)
                    if t1.shape[0] > 0 and t2.shape[0] > 0:
                        with torch.no_grad():
                            v1_d1 = self.df1(v1_missing_latent_eval, t1)
                            v2_d2 = self.df2(v2_missing_latent_eval, t2)
                        v2_recov = self.noise_scheduler.step(v2_d2, t2[0], v2_missing_latent_eval)
                        v1_recov = self.noise_scheduler.step(v1_d1, t1[0], v1_missing_latent_eval)
                if v1_recov.size(0) > 0 and v2_recov.size(0) > 0 and z_view2_both.size(
                        0) > 0 and z_view1_both.size(0) > 0:
                    ce_loss = criterion_instance(v1_recov, z_view2_both) + criterion_instance(v2_recov,
                                                                                              z_view1_both)
                else:
                    ce_loss = torch.tensor(0.0).to(device)
                loss =  rec_loss+ 1 * (dfloss + ce_loss) +  1 * mmi_loss +  1 * (cluster_loss + hc_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_df += dfloss.item()
                loss_mmi += mmi_loss.item()
                loss_cluster += cluster_loss.item()
                loss_hc += hc_loss.item()
                loss_ce += ce_loss.item()
            if (epoch) % config['print_num'] == 0:
                output = "Epoch: {:.0f}/{:.0f} " \
                         "==> loss = {:.4f} " .format(epoch, config['training']['epoch'], loss_all)
                print(output)
                scores = self.evaluation(config, mask, x1_train, x2_train, Y_list, device)
                accs.append(scores['accuracy'])
                nmis.append(scores['NMI'])
                aris.append(scores['ARI'])
                if scores['accuracy'] >= Tmp_acc:
                    Tmp_acc = scores['accuracy']
                    Tmp_nmi = scores['NMI']
                    Tmp_ari = scores['ARI']
        return Tmp_acc, Tmp_nmi, Tmp_ari

    def evaluation(self, config, mask, x1_train, x2_train, Y_list, device):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.df1.eval(), self.df2.eval()
            v1_all = mask[:, 0] == 1
            v2_all = mask[:, 1] == 1
            v1_missing = mask[:, 0] == 0
            v2_missing = mask[:, 1] == 0
            v1_embed = self.autoencoder1.encoder(x1_train[v1_all])
            v2_embed = self.autoencoder2.encoder(x2_train[v2_all])
            latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                device)
            latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                device)
            if x2_train[v1_missing].shape[0] != 0:
                v1_missing_latent_eval = self.autoencoder2.encoder(x2_train[v1_missing])
                v2_missing_latent_eval = self.autoencoder1.encoder(x1_train[v2_missing])
                timesteps = list(range(len(self.noise_scheduler)))[::-1]
                for i, t in enumerate(timesteps):
                    t1 = torch.from_numpy(np.repeat(t, v1_missing_latent_eval.shape[0])).long().to(device)
                    t2 = torch.from_numpy(np.repeat(t, v2_missing_latent_eval.shape[0])).long().to(device)
                    with torch.no_grad():
                        v1_d1 = self.df1(v1_missing_latent_eval, t1)
                        v2_d2 = self.df2(v2_missing_latent_eval, t2)
                    v2_recov = self.noise_scheduler.step(v2_d2, t2[0], v2_missing_latent_eval)
                    v1_recov = self.noise_scheduler.step(v1_d1, t1[0], v1_missing_latent_eval)
                latent_code_img_eval[v1_missing] = v1_recov
                latent_code_txt_eval[v2_missing] = v2_recov
            latent_code_img_eval[v1_all] = v1_embed
            latent_code_txt_eval[v2_all] = v2_embed
            latent_fusion = self.AttentionLayer(latent_code_img_eval, latent_code_txt_eval)
            y, _ = self.clusterLayer(latent_fusion)
            y = y.data.cpu().numpy().argmax(1)
            scores = evaluation(y_pred=y, y_true=Y_list[0])
            self.autoencoder1.train(), self.autoencoder2.train()
            self.df1.train(), self.df2.train()

        return scores
