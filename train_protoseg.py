import sys

from skimage import io

from torch import optim
from torch.backends import cudnn
from torch.nn import functional

from config import Config
from data import list_dataset, list_loader
from models import *
from utils import *

cudnn.benchmark = True

settings = Config()
general_params = settings['GENERAL']
fewshot_params = settings['FEW-SHOT']
task_dicts = settings['TASKS']['task_dicts']
args = settings['TRAINING']
datainfo = settings['DATA']

# Predefining directories.
ckpt_path = general_params['ckpt_path']
outp_path = general_params['outp_path']

# Reading system parameters.
conv_name = general_params['model']
data_name = general_params['dataset']
task_name = general_params['task']
fold_name = general_params['fold']

# Sparsity parameters.
listLoader = list_loader.ListLoader(fewshot_params)

# Setting experiment name.
exp_name = 'protoseg_multiple_' + conv_name + '_' + data_name + '_' + task_name + '_f' + str(fold_name)


# Function to get the number of annotated pixels for each class
def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)

    with torch.no_grad():
        num_samples = targets.new_zeros((batch_size, num_classes), dtype=dtype)

        for i in range(batch_size):
            trg_i = targets[i]

            for c in range(num_classes):
                num_samples[i, c] += trg_i[trg_i == c].size(0)

    return num_samples


def get_prototypes(embeddings, targets, num_classes):
    """Compute the prototypes (the mean vector of the embedded training/support
    points belonging to its class) for each classes in the task.
    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor 
        has shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has 
        shape `(batch_size, num_examples)`.
    num_classes : int
        Number of classes in the task.
    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)

    num_samples = get_num_samples(targets, num_classes, dtype=embeddings.dtype)
    num_samples.unsqueeze_(-1)
    num_samples = torch.max(num_samples, torch.ones_like(num_samples))

    prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
    indices = targets.unsqueeze(-1).expand_as(embeddings)

    for i in range(indices.size(0)):
        trg_i = targets[i]
        emb_i = embeddings[i]

        for c in range(num_classes):
            prototypes[i, c] += torch.sum(emb_i[trg_i == c], dim=0)

    prototypes.div_(num_samples)

    return prototypes


def prototypical_loss(prototypes, embeddings, targets, **kwargs):
    """Compute the loss (i.e. negative log-likelihood) for the prototypical 
    network, on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(batch_size, num_examples)`.
    Returns
    -------
    loss : `torch.FloatTensor` instance
        The negative log-likelihood on the query points.
    """
    squared_distances = torch.sum((prototypes.unsqueeze(2)
                                   - embeddings.unsqueeze(1)) ** 2, dim=-1)
    return functional.cross_entropy(-squared_distances, targets, **kwargs)


def get_predictions(prototypes, embeddings):
    """Compute the accuracy of the prototypical network on the test/query points.
    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
                              - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)
    return predictions


# Main function.
def main():

    # Setting network architecture.
    net = UNet(datainfo['num_channels'], datainfo['num_class'])
    if args["use_gpu"]:
        net = net.cuda()

    print(net)
    
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('# of parameters: ' + str(n_params))
    sys.stdout.flush()
    
    resize_to = (args['h_size'], args['w_size'])
    
    # Setting meta datasets.
    print('Setting meta-dataset loaders...')
    sys.stdout.flush()
    # meta_train_set = [list_dataset.ListDataset('meta_train',
    #                                            d['domain'],
    #                                            d['task'],
    #                                            fold_name,
    #                                            resize_to,
    #                                            num_shots=-1,
    #                                            sparsity_mode='random',
    #                                            imgtype=datainfo['imgtype'])
    #                   for d in task_dicts if d['domain'] != data_name or d['task'] != task_name]
    # meta_test_set = [list_dataset.ListDataset('meta_test',
    #                                           d['domain'],
    #                                           d['task'],
    #                                           fold_name,
    #                                           resize_to,
    #                                           num_shots=-1,
    #                                           sparsity_mode='dense',
    #                                           imgtype=datainfo['imgtype'])
    #                  for d in task_dicts if d['domain'] != data_name or d['task'] != task_name]
    meta_train_set = [list_dataset.ListDataset('meta_train',
                                               d['domain'],
                                               d['task'],
                                               fold_name,
                                               resize_to,
                                               num_shots=-1,
                                               sparsity_mode='random',
                                               imgtype=datainfo['imgtype'])
                      for d in task_dicts]
    meta_test_set = [list_dataset.ListDataset('meta_test',
                                              d['domain'],
                                              d['task'],
                                              fold_name,
                                              resize_to,
                                              num_shots=-1,
                                              sparsity_mode='dense',
                                              imgtype=datainfo['imgtype'])
                     for d in task_dicts]

    # Setting tuning and testing loaders.
    print('Setting tuning loaders...')
    sys.stdout.flush()
    loader_dict = listLoader.get_loaders(
        data_name,
        task_name,
        fold_name,
        resize_to,
        args, imgtype=datainfo['imgtype'])
    
    # Setting optimizer.
    meta_optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], betas=(args['momentum'], 0.99))
    
    # Setting scheduler.
    scheduler = optim.lr_scheduler.StepLR(meta_optimizer,
                                          args['lr_scheduler_step_size'],
                                          gamma=args['lr_scheduler_gamma'],
                                          last_epoch=-1)

    # Loading optimizer state in case of resuming training.
    if args['snapshot'] == '':
        curr_epoch = 1

    else:
        print('Training resuming from epoch ' + str(args['snapshot']) + '...')
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'meta.pth')))
        meta_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_meta.pth')))
        curr_epoch = int(args['snapshot']) + 1
    
    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))

    # Iterating over epochs.
    for epoch in range(curr_epoch, args['epoch_num'] + 1):

        # Meta training on source datasets.
        meta_train_test(meta_train_set,
                        meta_test_set,
                        net,
                        meta_optimizer,
                        epoch,
                        epoch % args['test_freq'] == 0)
        
        if epoch % args['test_freq'] == 0:
            
            run_sparse_tuning(loader_dict,
                              net,
                              epoch)
            
        scheduler.step()


# Training function.
def meta_train_test(meta_train_set, meta_test_set, net, meta_optimizer, epoch, save_model):
    
    # Setting network for training mode.
    net.train()
    
    # List for batch losses.
    train_outer_loss_list = list()
    
    num_tasks = len(meta_train_set)
    
    n_batches = 5
    
    # Iterating over batches.
    for i in range(n_batches):
        
        # Randomly selecting tasks.
        perm = np.random.permutation(num_tasks)
        print('Ep: ' + str(epoch) + ', it: ' + str(i + 1) + ', task subset: ' + str(perm[:args['n_metatasks_iter']]))
        sys.stdout.flush()

        indices = perm[:args['n_metatasks_iter']]
        
        for index in indices:
            # Acquiring training and test data.
            x_train = []
            y_train = []
            
            x_test = []
            y_test = []
            
            x_tr, y_tr, x_ts, y_ts = prepare_meta_batch(meta_train_set,
                                                        meta_test_set,
                                                        index,
                                                        args['batch_size'],
                                                        args['use_gpu'])
            
            x_train.append(x_tr)
            y_train.append(y_tr)
            
            x_test.append(x_ts)
            y_test.append(y_ts)
        
            # Concatenating tensors.
            x_train = torch.cat(x_train, dim=0)
            y_train = torch.cat(y_train, dim=0)
            
            x_test = torch.cat(x_test, dim=0)
            y_test = torch.cat(y_test, dim=0)
            
            # Clearing model gradients.
            net.zero_grad()
            
            ##########################################################################
            # Start of prototyping. ##################################################
            ##########################################################################
            
            emb_train = net(x_train)
            emb_test = net(x_test)
            
            emb_train_linear = emb_train.permute(0, 2, 3, 1).view(
                emb_train.size(0), emb_train.size(2) * emb_train.size(3), emb_train.size(1))
            emb_test_linear = emb_test.permute(0, 2, 3, 1).view(
                emb_test.size(0), emb_test.size(2) * emb_test.size(3), emb_test.size(1))
            
            y_train_linear = y_train.view(y_train.size(0), -1)
            y_test_linear = y_test.view(y_test.size(0), -1)
            
            prototypes = get_prototypes(emb_train_linear,
                                        y_train_linear,
                                        datainfo['num_class'])
            
            outer_loss = prototypical_loss(prototypes,
                                           emb_test_linear,
                                           y_test_linear,
                                           ignore_index=-1)
                
            ##########################################################################
            # End of prototyping. ####################################################
            ##########################################################################
            
            # Clears the gradients of meta_optimizer.
            meta_optimizer.zero_grad()
            
            # Computing backpropagation.
            outer_loss.backward()
            meta_optimizer.step()
            
            # Updating loss meter.
            train_outer_loss_list.append(outer_loss.detach().item())
    
    # Saving meta-model.
    if save_model:
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'meta.pth'))
        torch.save(meta_optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_meta.pth'))

    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [train loss %.4f]' % (
        epoch, np.asarray(train_outer_loss_list).mean()))
    print('--------------------------------------------------------------------')
    sys.stdout.flush()
    
    
def run_sparse_tuning(loader_dict, net, epoch):
    
    # Tuning/testing on points.
    for dict_points in loader_dict['points']:

        n_shots = dict_points['n_shots']
        sparsity = dict_points['sparsity']

        print('    Evaluating \'points\' (%d-shot, %d-points)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_points['train'],
                        dict_points['test'],
                        net,
                        epoch,
                        'points_(%d-shot_%d-points)' % (n_shots, sparsity))

    # Tuning/testing on contours.
    for dict_contours in loader_dict['contours']:

        n_shots = dict_contours['n_shots']
        sparsity = dict_contours['sparsity']

        print('    Evaluating \'contours\' (%d-shot, %.2f-density)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_contours['train'],
                        dict_contours['test'],
                        net,
                        epoch,
                        'contours_(%d-shot_%.2f-density)' % (n_shots, sparsity))

    # Tuning/testing on grid.
    for dict_grid in loader_dict['grid']:

        n_shots = dict_grid['n_shots']
        sparsity = dict_grid['sparsity']

        print('    Evaluating \'grid\' (%d-shot, %d-spacing)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_grid['train'],
                        dict_grid['test'],
                        net,
                        epoch,
                        'grid_(%d-shot_%d-spacing)' % (n_shots, sparsity))

    # Tuning/testing on regions.
    for dict_regions in loader_dict['regions']:

        n_shots = dict_regions['n_shots']
        sparsity = dict_regions['sparsity']

        print('    Evaluating \'regions\' (%d-shot, %.2f-regions)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_regions['train'],
                        dict_regions['test'],
                        net,
                        epoch,
                        'regions_(%d-shot_%.2f-regions)' % (n_shots, sparsity))

    # Tuning/testing on skels.
    for dict_skels in loader_dict['skels']:

        n_shots = dict_skels['n_shots']
        sparsity = dict_skels['sparsity']

        print('    Evaluating \'skels\' (%d-shot, %.2f-skels)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_skels['train'],
                        dict_skels['test'],
                        net,
                        epoch,
                        'skels_(%d-shot_%.2f-skels)' % (n_shots, sparsity))

    # Tuning/testing on dense.
    for dict_dense in loader_dict['dense']:

        n_shots = dict_dense['n_shots']

        print('    Evaluating \'dense\' (%d-shot)...' % n_shots)
        sys.stdout.flush()

        tune_train_test(dict_dense['train'],
                        dict_dense['test'],
                        net,
                        epoch,
                        'dense_(%d-shot)' % n_shots)


def tune_train_test(tune_train_loader, tune_test_loader, net, epoch, sparsity_mode):
    
    # Creating output directories.
    if epoch == args['epoch_num']:
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch)))
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_test_epoch_' + str(epoch)))
    
    with torch.no_grad():
    
        # Setting network for training mode.
        net.eval()

        # Zeroing model gradient.
        net.zero_grad()

        # Creating lists for tune train embeddings and labels.
        emb_train_list = []
        y_train_list = []

        # Iterating over tuning train batches.
        for i, data in enumerate(tune_train_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            x_tr, _, y_tr, _ = data

            # Casting tensors to cuda.
            if args["use_gpu"]:
                x_tr = x_tr.cuda()
                y_tr = y_tr.cuda()

            emb_tr = net(x_tr)

            emb_train_linear = emb_tr.permute(0, 2, 3, 1).view(
                emb_tr.size(0), emb_tr.size(2) * emb_tr.size(3), emb_tr.size(1))
            
            y_train_linear = y_tr.view(y_tr.size(0), -1)

            emb_train_list.append(emb_train_linear)
            y_train_list.append(y_train_linear)

        emb_tr = torch.vstack(emb_train_list)
        y_tr = torch.vstack(y_train_list)
        
        prototypes = get_prototypes(emb_tr, y_tr, datainfo['num_class'])

        # Lists for whole epoch loss.
        labs_all, prds_all = [], []
        
        # Iterating over tuning test batches.
        for i, data in enumerate(tune_test_loader):
            # Obtaining images, dense labels, sparse labels and paths for batch.
            x_ts, y_ts, _, img_name = data
            
            # Casting tensors to cuda.
            if args["use_gpu"]:
                x_ts = x_ts.cuda()
                y_ts = y_ts.cuda()
            
            emb_ts = net(x_ts)
            
            emb_test_linear = emb_ts.permute(0, 2, 3, 1).view(
                emb_ts.size(0), emb_ts.size(2) * emb_ts.size(3), emb_ts.size(1))
            
            p_test_linear = get_predictions(prototypes, emb_test_linear)
            
            p_test = p_test_linear.view(p_test_linear.size(0), y_ts.size(1), y_ts.size(2))
            
            # Taking mode of predictions.
            p_full, _ = torch.mode(p_test, dim=0)
            
            labs_all.append(y_ts.cpu().numpy().squeeze())
            prds_all.append(p_full.cpu().numpy().squeeze())

            # Saving predictions.
            if epoch == args['epoch_num']:
                stored_pred = p_full.cpu().numpy().squeeze()
                stored_pred = (stored_pred * (255 / stored_pred.max())).astype(np.uint8)
                io.imsave(
                    os.path.join(outp_path, exp_name, sparsity_mode + '_test_epoch_' + str(epoch),
                                 img_name[0] + '.png'),
                    stored_pred)
            
    # Converting to numpy for computing metrics.
    labs_np = np.asarray(labs_all).ravel()
    prds_np = np.asarray(prds_all).ravel()

    # Computing metrics.
    iou = metrics.jaccard_score(labs_np, prds_np)

    # Printing metric.
    print('--------------------------------------------------------------------')
    print('Jaccard test "%s": %.2f' % (sparsity_mode, iou * 100))
    print('--------------------------------------------------------------------')

    if epoch == args['epoch_num']:

        # Iterating over tuning train batches for saving.
        for i, data in enumerate(tune_train_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            _, y_dense, y_sparse, img_name = data

            for j in range(len(img_name)):
                stored_dense = y_dense[j].cpu().squeeze().numpy()
                stored_dense = (stored_dense * (255 / stored_dense.max())).astype(np.uint8)
                stored_sparse = y_sparse[j].cpu().squeeze().numpy() + 1
                stored_sparse = (stored_sparse * (255 / stored_sparse.max())).astype(np.uint8)
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch),
                                       img_name[j] + '_dense.png'),
                          stored_dense)
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch), 
                                       img_name[j] + '_sparse.png'),
                          stored_sparse)
                

if __name__ == '__main__':
    main()
