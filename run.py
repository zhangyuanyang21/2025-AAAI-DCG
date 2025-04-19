import argparse
import itertools
from get_indicator_matrix_A import get_mask
from datasets import *
from configure import get_default_config
from ICDM import *

def main(MR=[0.3]):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    config = get_default_config(dataset)
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    config['print_num'] = 1
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[1]

    for missingrate in MR:
        config['training']['missing_rate'] = missingrate
        print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
        for data_seed in range(1, args.test_time + 1):
            np.random.seed(1)
            mask = get_mask(2, x1_train_raw.shape[0], config['training']['missing_rate'])
            # mask the data
            x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
            x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

            x1_train = torch.from_numpy(x1_train).float().to(device)
            x2_train = torch.from_numpy(x2_train).float().to(device)
            mask = torch.from_numpy(mask).long().to(device)
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            # Build the model
            ICDM = icdm(config)
            optimizer = torch.optim.Adam(
                itertools.chain(ICDM.autoencoder1.parameters(), ICDM.autoencoder2.parameters(), ICDM.df1.parameters(), ICDM.df2.parameters(), ICDM.clusterLayer.parameters(),ICDM.AttentionLayer.parameters()),
                lr=config['training']['lr'])
            ICDM.to_device(device)
            # Training
            acc, nmi, ari = ICDM.train(config, x1_train, x2_train, Y_list, mask, optimizer, device)
            print('-------------------The ' + str(data_seed) + ' training over Missing rate = ' + str(missingrate) + '--------------------')
            print("ACC {:.2f}, NMI {:.2f}, ARI {:.2f}".format(acc, nmi, ari))

if __name__ == '__main__':
    dataset = {
               1: "LandUse_21",
               2: "CUB",
               3: "HandWritten",
               4: "Multi-Fashion",
               5: 'Synthetic3d',
               }

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(2), help='dataset id')  # data index
    parser.add_argument('--test_time', type=int, default=str(1), help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.3]

    main(MR=MisingRate)
