def get_default_config(data_name):
    if data_name in ['CUB']:
        """The default configs."""
        return dict(
            Autoencoder=dict(
                arch1=[1024, 512, 1024, 1024, 128], 
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
                    n_clusters=3,
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
                num_timesteps=100,
                beta_schedule="linear",
            ),
        )
    
    else:
        raise Exception('Undefined data_name')
