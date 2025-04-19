def get_default_config(data_name):
    if data_name in ['Scene-15']:
        return dict(
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 16],
                arch2=[20, 1024, 1024, 1024, 16],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=16,
                mask_seed=4,
                batch_size=512,
                epoch=200,
                lr=1.0e-4,
                lambda1=10,
                lambda2=0.1,
                n_clusters=15,
            ),
            diffusion=dict(
                emb_size=16,
                time_type="sinusoidal",
                out_dim1=16,
                out_dim2=16,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['CCV']:
        return dict(
            Autoencoder=dict(
                arch1=[5000, 1024, 1024, 1024, 128],
                arch2=[5000, 1024, 1024, 1024, 128],
                # arch2=[4000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=20,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Cora']:
        return dict(
            Autoencoder=dict(
                arch1=[2708, 1024, 1024, 1024, 128],
                arch2=[1433, 1024, 1024, 1024, 128],
                # arch2=[2048, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=7,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['aloideep3v']:
        return dict(
            Autoencoder=dict(
                arch1=[2048, 1024, 1024, 1024, 128],
                arch2=[4096, 1024, 1024, 1024, 128],
                # arch2=[2048, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=100,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Synthetic3d']:
        return dict(
            Autoencoder=dict(
                arch1=[3, 1024, 1024, 1024, 128],
                arch2=[3, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=3,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps= 100,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['SUNRGBD_fea']:
        return dict(
            Autoencoder=dict(
                arch1=[4096, 1024, 1024, 1024, 128],
                arch2=[4096, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=45,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['NH_p4660']:
        return dict(
            Autoencoder=dict(
                arch1=[6750, 1024, 1024, 1024, 128],
                arch2=[3304, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=5,#改
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Caltech101-20']:
        return dict(
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=1024,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=20,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['NGs']:
        return dict(
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 128],
                arch2=[2000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=5,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 32],
                arch2=[784, 1024, 1024, 1024, 32],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=2,
                mask_seed=5,
                batch_size=512,
                epoch=200,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=32,
                time_type="sinusoidal",
                out_dim1=32,
                out_dim2=32,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['HandWritten']:
        return dict(
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[64, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=200,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Caltech-5V']:
        return dict(
            Autoencoder=dict(
                arch1=[512, 1024, 1024, 1024, 128],
                arch2=[928, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=7,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Multi-Fashion']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=2,
                mask_seed=5,
                batch_size=256,
                epoch=200,
                lr=1.0e-4,
                lambda1=10.0,
                lambda2=0.1,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Reuters']:
        return dict(
            Autoencoder=dict(
                arch1=[10, 1024, 1024, 1024, 128],
                arch2=[10, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=6,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=200,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['MSRC_v1']:
        return dict(
            Autoencoder=dict(
                arch1=[24, 1024, 1024, 1024, 128],
                arch2=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=6,
                mask_seed=5,
                missing_rate=0.3,
                batch_size=256,
                epoch=500,
                lr=1.0e-4,
                lambda1=1,
                lambda2=0.1,
                n_clusters=7,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['Hdigit']:
        """The default configs."""
        return dict(
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[256, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                mask_seed=5,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                temperature_f=0.5,
                temperature_l=1,
                n_clusters=10,
            ),
        )
    # 效果不好
    elif data_name in ['CUB']:#feature
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[1024, 512, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[300, 512, 1024, 1024, 128],
                activations1='sigmoid',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                mask_seed=5,
                batch_size=256,
                epoch=200,
                lr=1e-4,
                num=10,
                dim=256,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=100,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['MNIST-USPS']:
        return dict(
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 128],
                arch2=[784, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=0,
                mask_seed=5,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                lambda1=100,
                lambda2=0.1,
                n_clusters=10,
            ),
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    elif data_name in ['BDGP']:
        """The default configs."""
        return dict(
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
            Autoencoder=dict(
                arch1=[1750, 1024, 1024, 1024, 128],
                arch2=[79, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=8,
                mask_seed=5,
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                class_num=10,
                temperature_f=0.5,
                temperature_l=1,
                n_clusters=5,
            ),
        )
    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            diffusion=dict(
                emb_size=128,
                time_type="sinusoidal",
                out_dim1=128,
                out_dim2=128,
            ),
            noise_scheduler=dict(
                num_timesteps=100,
                beta_schedule="linear",
            ),
            Autoencoder=dict(
                arch1=[59, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.5,
                seed=3,
                mask_seed=5,
                epoch=200,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                temperature_f=0.5,
                temperature_l=1,
                n_clusters=21,
            ),
        )
    elif data_name in ['RGB-D']:
        return dict(
            Autoencoder=dict(
                arch1=[300, 1024, 1024, 1024, 256],
                arch2=[2048, 1024, 1024, 1024, 256],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0.3,
                seed=8,
                mask_seed=5,
                batch_size=256,
                epoch=2000,
                lr=1.0e-4,
                lambda1=10,
                lambda2=1,
                n_clusters=13,
            ),
            diffusion=dict(
                emb_size=256,
                time_type="sinusoidal",
                out_dim1=256,
                out_dim2=256,
            ),
            noise_scheduler=dict(
                num_timesteps=50,
                beta_schedule="linear",
            ),
        )
    else:
        raise Exception('Undefined data_name')
