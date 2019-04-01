import argparse
from torch.backends import cudnn
cudnn.enabled = False
import scipy.io as sio


def main(config):
    # svhn_loader, mnist_loader = get_loader(config)

    svhn_loader, mnist_loader = None, None
    from MAN import Solver
    solver = Solver(config, svhn_loader, mnist_loader)
    cudnn.benchmark = True

    results = solver.train()
    # if config.just_valid:
    #     (best_valid_results, discriminator_losses, generator_losses, fisher_losses, valid_results, train_features_list, valid_features_list, test_features_list) = results
    #     np.save('Convergence/Convergence_' + config.datasets + '_Epoch' + str(config.epochs) + '_' + config.strategy + '.npy', {'best_valid_results': best_valid_results, 'discriminator_losses': discriminator_losses, 'generator_losses': generator_losses, 'fisher_losses': fisher_losses, 'valid_results': valid_results, 'train_features_list': train_features_list, 'valid_features_list': valid_features_list, 'test_features_list': test_features_list})
    # else:
    #     sio.savemat('Results_MAN/params_' + config.datasets + '_' + str(config.epochs) + '_final_resutls.mat', {'param_results': np.array(results)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--num_classes', type=int, default=-1)
    parser.add_argument('--text_mode', type=str, default='multichannel')
    # parser.add_argument('--text_mode', type=str, default='static')

    # training hyper-parameters
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # misc
    parser.add_argument('--compute_all', type=bool, default=False)
    parser.add_argument('--just_valid', type=bool, default=False) # wiki, pascal, nus-wide, xmedianet


    parser.add_argument('--ALL', type=bool, default=False)
    parser.add_argument('--fisher_beta', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=256) # 256
    parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_shape', type=int, default=-1)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--datasets', type=str, default='xmedia') # wiki, pascal, reuters, xmedia, xmedia_pairwise
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--sample_interval', type=int, default=1)

    config = parser.parse_args()
    print(config)
    main(config)
