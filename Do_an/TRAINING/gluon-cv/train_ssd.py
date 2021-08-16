"""Train SSD"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import glob
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.data.transforms.presets.ssd import SSDDALIPipeline
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy

from mxnet.contrib import amp

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    from nvidia.dali.plugin.mxnet import DALIGenericIterator
    dali_found = True
except ImportError:
    dali_found = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--dataset-root', type=str, default='/content/CUSTOM',
                        help='Path of the directory where the dataset is located.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--dali', action='store_true',
                        help='Use DALI for data loading and data preprocessing in training. '
                        'Currently supports only COCO.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                        '--gpus is ignored when using --horovod.')

    args = parser.parse_args()
    if args.horovod:
        assert hvd, "You are trying to use horovod support but it's not installed"
    return args
from gluoncv.data import VOCDetection
class VOCLike(VOCDetection):
    #CLASSES = ['bocuoi_bich']
    CLASSES = ['coloscare_hong', 'coloscare_vang', 'sumi_orange', 'sumi_cherry', 'coloscare_xanh', 'smarta_grow', 'tobokki_vang', 'keo_kimiko', 'vansua_nestle', 'vansua_burine', 'rongbien_bbq', 'poro_do', 'poro_xanhla', 'kun_tim_suabich', 'kun_cam_suabich', 'poro_tim', 'poro_xanhdatroi', 'hipp_buckwheat', 'tobokki_do', 'kun_hong_suabich', 'kun_do_suabich', 'kun_xanhla_suabich', 'aptamil', 'hanie_kid', 'metacare_blue', 'coloscare_hong1', 'ostelin_kids', 'hd_kids', 'nutricare_bone', 'kazu_hong', 'leankid_100_tim', 'nestle_nan_xanhtroi', 'kazu_xanh', 'kazu_vang', 'botandam_ridielac_xanhtroi', 'suanon_goldilac', 'botandam_metacare_hong', 'botandam_metacare_xanh', 'bot_andam_metacare_vang', 'mamamy_daugoi', 'Lactacycl_bb', 'gohmo_suatam_nhat', 'gohmo_suatam_dam', 'cetaphil', 'pureen_suatam_trang', 'dnee_kid_do', 'dnee_kid_xanh', 'dr_smile', 'diepannhi', 'pureen_vang', 'ro_luoi', 'nuocmuoi', 'botandam_dielac_do', 'benokid_colos', 'vinamilk_suabich', 'fami_suabich', 'pentavite', 'nunest_hop', 'beno_kid', 'milktea_hong', 'matcha_vang', 'kazu_xanhla', 'khautrang_kid', 'elemis_xanh', 'ongtho_hop', 'suaoccho_nau', 'care_100_gold_trang', 'metacare_gold_xanhtroi', 'metacare_gold_xanhla', 'metacare_gold_hong', 'care_100_trang_xanh', 'goigiavi_xanhla', 'goigiavi_do', 'goigiavi_nau', 'meji_hong', 'meji_nau', 'metamom_tim', 'suaoccho_1hop', 'care100_xanhla_1loc', 'hero_cam_1loc', 'pediasure_xanhhong_1loc', 'growplus_do_1loc', 'leankid100_vang_1loc', 'smarta_vang_1loc', 'babyme_1loc', 'kazu_hong_1loc', 'enfagrow_1loc', 'kazu_xanh_1loc', 'Th_1loc', 'yoko_1loc', 'kun_do_1loc', 'grow_vang_1loc', 'care100_vang_1loc', 'optimum_gold_1loc', 'coloscare_trangxanh_1loc', 'vinamilk_itduong_1loc', 'grow_xanhtrang_1loc', 'metacare_xanhdam_1loc', 'alponte_1loc', 'dalatmilk_1loc', 'drluxia_dovang_1loc', 'khautrang_hopnho', 'botngucoc_nau', 'dau_chongmuoi', 'bocuoi_tron', 'bocuoi_bich', 'cheese_dessert_hong', 'cheese_dessert_xanh', 'goi_mau_vang', 'chocopie_hop', 'nuocyen_1loc', 'poro_bichxanh', 'poro_bichhong', 'chao_toyen_vang', 'chaotuoi_vang', 'chaotuoi_do', 'chaotuoi_xanhla', 'chaotuoi_hong', 'babyme_nest', 'vansua_bledina_hong', 'vansua_bledina_vang', 'suachua_hoff_xanhvang', 'suachua_hoff_vang', 'suachua_hoff_xanhtroi', 'suachua_hoff_do', 'khanuot_embe_do', 'khanuot_embe_xanhtroi', 'khanuot_embe_xanhla', 'khanuot_embe_vang', 'khanuot_trangsocxanh', 'tinhdau_xanhla', 'tinhdau_trang', 'ruatay_lifeboy', 'ruatay_24_vang', 'ruatay_suzu_xanh', 'ruatay_suzu_hong', 'hamsua_hong', 'hamsua_xanh', 'ensure_1loc', 'breastmilk_hong', 'binhsua_hongdo', 'breast_milk_xanh', 'breast_milk_trang_xanh', 'phan_pureen_trang', 'ta_popolin_trang', 'super_unidry_vangm28', 'bobby_40+4', 'huggies_dry_L', 'bobby_+6', 'bobby_extrasofting_L', 'binhsua_baby_xanhtroi', 'binhsua_baby_trangcao', 'binhsua_baby_hong', 'binhsua_baby_xanhla', 'binhsua_baby_vang', 'ho_trangxanh_hop', 'DHA_hongvang', 'dauphong_chai', 'banh_hellokitty', 'dauoliu_vangnhat', 'kiddy_chaiden', 'mam_vio_cam', 'mam_vio_xanhnhat', 'dau_oliu_dintel', 'dau_oliu_chailun', 'babybites_vang', 'babybites_do', 'babybites_xanh', 'tambong_xanhla', 'susu_nau_1loc', 'susu_cam_1loc', 'kemdanhrang_vang', 'kemdanhrang_do', 'susu_xanhla_1loc', 'tambong_xanhtroi', 'tatawa_cappucino_do', 'chao_wakodo_hong', 'keo_morinaga_do', 'chao_wakodo_xanhduong', 'nuoinhat_hong_60k', 'bunnhat_cam_60k', 'banhgao_treem_xanhla', 'banhgao_treem_xanhduong', 'dau_ajinomoto', 'binhsua_pur_xanhduong', 'binhsua_babiboo_cam', 'binhsua_latan_do', 'binhsua_ucare_cam', 'botdanhrang_pur', 'numvu_pur', 'khanvai_hientrang', 'khanvai_babyquy_nho', 'num_latan_tim', 'num_latan_xanhduong', 'num_latan_xanhla', 'num_pur_xanhduong', 'khangiay_emos_tim', 'vinamilk_itduong_hopto', 'asutralian100_hopto', 'giat_dnee_xanhduong', 'giat_fineline_tim', 'giat_dnee_xanhduong2', 'giat_fineline_hong', 'giat_dnee_xanhla', 'lausan_lovic_hong', 'bovesinh_babiboo_cam', 'origold_xanhduong', 'origold_tim', 'origold_xanhla', 'origold_hon']
    for i in range(len(CLASSES)):
      CLASSES[i] = CLASSES[i].lower()
    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = VOCLike(root='/content/CUSTOM', splits=(('', 'train'),))
        val_dataset = VOCLike(root='/content/CUSTOM', splits=(('', 'val'),))
        print('Train: ' + str(len(train_dataset)))
        print('Valid: ' + str(len(val_dataset)))
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_train2017')
        val_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx))
    anchors = anchors.as_in_context(mx.cpu())
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def get_dali_dataset(dataset_name, devices, args):
    if dataset_name.lower() == "coco":
        # training
        expanded_file_root = os.path.expanduser(args.dataset_root)
        coco_root = os.path.join(expanded_file_root,
                                 'coco',
                                 'train2017')
        coco_annotations = os.path.join(expanded_file_root,
                                        'coco',
                                        'annotations',
                                        'instances_train2017.json')
        if args.horovod:
            train_dataset = [gdata.COCODetectionDALI(num_shards=hvd.size(), shard_id=hvd.rank(), file_root=coco_root,
                                                     annotations_file=coco_annotations, device_id=hvd.local_rank())]
        else:
            train_dataset = [gdata.COCODetectionDALI(num_shards= len(devices), shard_id=i, file_root=coco_root,
                                                     annotations_file=coco_annotations, device_id=i) for i, _ in enumerate(devices)]

        # validation
        if (not args.horovod or hvd.rank() == 0):
            val_dataset = gdata.COCODetection(root=os.path.join(args.dataset_root + '/coco'),
                                              splits='instances_val2017',
                                              skip_empty=False)
            val_metric = COCODetectionMetric(
                val_dataset, args.save_prefix + '_eval', cleanup=True,
                data_shape=(args.data_shape, args.data_shape))
        else:
            val_dataset = None
            val_metric = None
    else:
        raise NotImplementedError('Dataset: {} not implemented with DALI.'.format(dataset_name))

    return train_dataset, val_dataset, val_metric

def get_dali_dataloader(net, train_dataset, val_dataset, data_shape, global_batch_size, num_workers, devices, ctx, horovod, seed):
    width, height = data_shape, data_shape
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width), ctx=ctx))
    anchors = anchors.as_in_context(mx.cpu())

    if horovod:
        batch_size = global_batch_size // hvd.size()
        pipelines = [SSDDALIPipeline(device_id=hvd.local_rank(), batch_size=batch_size,
                                     data_shape=data_shape, anchors=anchors,
                                     num_workers=num_workers, dataset_reader = train_dataset[0],
                                     seed=seed)]
    else:
        num_devices = len(devices)
        batch_size = global_batch_size // num_devices
        pipelines = [SSDDALIPipeline(device_id=device_id, batch_size=batch_size,
                                     data_shape=data_shape, anchors=anchors,
                                     num_workers=num_workers,
                                     dataset_reader = train_dataset[i],
                                     seed=seed) for i, device_id in enumerate(devices)]

    epoch_size = train_dataset[0].size()
    if horovod:
        epoch_size //= hvd.size()
    train_loader = DALIGenericIterator(pipelines, [('data', DALIGenericIterator.DATA_TAG),
                                                    ('bboxes', DALIGenericIterator.LABEL_TAG),
                                                    ('label', DALIGenericIterator.LABEL_TAG)],
                                                    epoch_size, auto_reset=True)

    # validation
    if (not horovod or hvd.rank() == 0):
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(width, height)),
            global_batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    else:
        val_loader = None

    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix, args):
    current_map = float(current_map)
    print("Current mAP: " + str(current_map))
    print("Best mAP: " + str(best_map[0]))
    check = False
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        check = True
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
    if check:
        params = []
        params = glob.glob(args.resume.strip() + '/*.params')
        params.sort()
        if len(params) >= 5:
          print('Remove ' + params[0])
          #os.remove(params[0])

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()

def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
                        net.collect_params(), 'sgd',
                        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})
    else:
        trainer = gluon.Trainer(
                    net.collect_params(), 'sgd',
                    {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum},
                    update_on_kvstore=(False if args.amp else None))

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)

        for i, batch in enumerate(train_data):
            if args.dali:
                # dali iterator returns a mxnet.io.DataBatch
                data = [d.data[0] for d in batch]
                box_targets = [d.label[0] for d in batch]
                cls_targets = [nd.cast(d.label[1], dtype='float32') for d in batch]

            else:
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                if args.amp:
                    with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)

            if (not args.horovod or hvd.rank() == 0):
                local_batch_size = int(args.batch_size // (hvd.size() if args.horovod else 1))
                ce_metric.update(0, [l * local_batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
                if args.log_interval and not (i + 1) % args.log_interval:
                    name1, loss1 = ce_metric.get()
                    name2, loss2 = smoothl1_metric.get()
                    logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, args.batch_size/(time.time()-btic), name1, loss1, name2, loss2))
                btic = time.time()

        if (not args.horovod or hvd.rank() == 0):
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2))
            if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix, args)
            break

if __name__ == '__main__':
    args = parse_args()

    if args.amp:
        amp.init()

    if args.horovod:
        hvd.init()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, 'coco'))#args.dataset))
    args.save_prefix += net_name
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        net = get_model(net_name, pretrained=True, norm_layer=gluon.nn.BatchNorm)
        classes = ['coloscare_hong', 'coloscare_vang', 'sumi_orange', 'sumi_cherry', 'coloscare_xanh', 'smarta_grow', 'tobokki_vang', 'keo_kimiko', 'vansua_nestle', 'vansua_burine', 'rongbien_bbq', 'poro_do', 'poro_xanhla', 'kun_tim_suabich', 'kun_cam_suabich', 'poro_tim', 'poro_xanhdatroi', 'hipp_buckwheat', 'tobokki_do', 'kun_hong_suabich', 'kun_do_suabich', 'kun_xanhla_suabich', 'aptamil', 'hanie_kid', 'metacare_blue', 'coloscare_hong1', 'ostelin_kids', 'hd_kids', 'nutricare_bone', 'kazu_hong', 'leankid_100_tim', 'nestle_nan_xanhtroi', 'kazu_xanh', 'kazu_vang', 'botandam_ridielac_xanhtroi', 'suanon_goldilac', 'botandam_metacare_hong', 'botandam_metacare_xanh', 'bot_andam_metacare_vang', 'mamamy_daugoi', 'Lactacycl_bb', 'gohmo_suatam_nhat', 'gohmo_suatam_dam', 'cetaphil', 'pureen_suatam_trang', 'dnee_kid_do', 'dnee_kid_xanh', 'dr_smile', 'diepannhi', 'pureen_vang', 'ro_luoi', 'nuocmuoi', 'botandam_dielac_do', 'benokid_colos', 'vinamilk_suabich', 'fami_suabich', 'pentavite', 'nunest_hop', 'beno_kid', 'milktea_hong', 'matcha_vang', 'kazu_xanhla', 'khautrang_kid', 'elemis_xanh', 'ongtho_hop', 'suaoccho_nau', 'care_100_gold_trang', 'metacare_gold_xanhtroi', 'metacare_gold_xanhla', 'metacare_gold_hong', 'care_100_trang_xanh', 'goigiavi_xanhla', 'goigiavi_do', 'goigiavi_nau', 'meji_hong', 'meji_nau', 'metamom_tim', 'suaoccho_1hop', 'care100_xanhla_1loc', 'hero_cam_1loc', 'pediasure_xanhhong_1loc', 'growplus_do_1loc', 'leankid100_vang_1loc', 'smarta_vang_1loc', 'babyme_1loc', 'kazu_hong_1loc', 'enfagrow_1loc', 'kazu_xanh_1loc', 'Th_1loc', 'yoko_1loc', 'kun_do_1loc', 'grow_vang_1loc', 'care100_vang_1loc', 'optimum_gold_1loc', 'coloscare_trangxanh_1loc', 'vinamilk_itduong_1loc', 'grow_xanhtrang_1loc', 'metacare_xanhdam_1loc', 'alponte_1loc', 'dalatmilk_1loc', 'drluxia_dovang_1loc', 'khautrang_hopnho', 'botngucoc_nau', 'dau_chongmuoi', 'bocuoi_tron', 'bocuoi_bich', 'cheese_dessert_hong', 'cheese_dessert_xanh', 'goi_mau_vang', 'chocopie_hop', 'nuocyen_1loc', 'poro_bichxanh', 'poro_bichhong', 'chao_toyen_vang', 'chaotuoi_vang', 'chaotuoi_do', 'chaotuoi_xanhla', 'chaotuoi_hong', 'babyme_nest', 'vansua_bledina_hong', 'vansua_bledina_vang', 'suachua_hoff_xanhvang', 'suachua_hoff_vang', 'suachua_hoff_xanhtroi', 'suachua_hoff_do', 'khanuot_embe_do', 'khanuot_embe_xanhtroi', 'khanuot_embe_xanhla', 'khanuot_embe_vang', 'khanuot_trangsocxanh', 'tinhdau_xanhla', 'tinhdau_trang', 'ruatay_lifeboy', 'ruatay_24_vang', 'ruatay_suzu_xanh', 'ruatay_suzu_hong', 'hamsua_hong', 'hamsua_xanh', 'ensure_1loc', 'breastmilk_hong', 'binhsua_hongdo', 'breast_milk_xanh', 'breast_milk_trang_xanh', 'phan_pureen_trang', 'ta_popolin_trang', 'super_unidry_vangm28', 'bobby_40+4', 'huggies_dry_L', 'bobby_+6', 'bobby_extrasofting_L', 'binhsua_baby_xanhtroi', 'binhsua_baby_trangcao', 'binhsua_baby_hong', 'binhsua_baby_xanhla', 'binhsua_baby_vang', 'ho_trangxanh_hop', 'DHA_hongvang', 'dauphong_chai', 'banh_hellokitty', 'dauoliu_vangnhat', 'kiddy_chaiden', 'mam_vio_cam', 'mam_vio_xanhnhat', 'dau_oliu_dintel', 'dau_oliu_chailun', 'babybites_vang', 'babybites_do', 'babybites_xanh', 'tambong_xanhla', 'susu_nau_1loc', 'susu_cam_1loc', 'kemdanhrang_vang', 'kemdanhrang_do', 'susu_xanhla_1loc', 'tambong_xanhtroi', 'tatawa_cappucino_do', 'chao_wakodo_hong', 'keo_morinaga_do', 'chao_wakodo_xanhduong', 'nuoinhat_hong_60k', 'bunnhat_cam_60k', 'banhgao_treem_xanhla', 'banhgao_treem_xanhduong', 'dau_ajinomoto', 'binhsua_pur_xanhduong', 'binhsua_babiboo_cam', 'binhsua_latan_do', 'binhsua_ucare_cam', 'botdanhrang_pur', 'numvu_pur', 'khanvai_hientrang', 'khanvai_babyquy_nho', 'num_latan_tim', 'num_latan_xanhduong', 'num_latan_xanhla', 'num_pur_xanhduong', 'khangiay_emos_tim', 'vinamilk_itduong_hopto', 'asutralian100_hopto', 'giat_dnee_xanhduong', 'giat_fineline_tim', 'giat_dnee_xanhduong2', 'giat_fineline_hong', 'giat_dnee_xanhla', 'lausan_lovic_hong', 'bovesinh_babiboo_cam', 'origold_xanhduong', 'origold_tim', 'origold_xanhla', 'origold_hon']
        #classes = ['bocuoi_bich']
        for i in range(len(classes)):
          classes[i] = classes[i].lower()
        net.reset_class(classes)
        async_net = net
    if args.resume.strip():
        a = glob.glob(args.resume.strip() + '/*.params')
        a.sort()
        print('Restore: ' + a[-2])
        args.start_epoch = int(a[-2].split('_')[-2]) + 1

        net.load_parameters(a[-2])#args.resume.strip())
        net.collect_params().reset_ctx(ctx[0])
        async_net.load_parameters(a[-2])#args.resume.strip())
        async_net.collect_params().reset_ctx(ctx[0])
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
            # needed for net to be first gpu when using AMP
            net.collect_params().reset_ctx(ctx[0])

    # training data
    if args.dali:
        if not dali_found:
            raise SystemExit("DALI not found, please check if you installed it correctly.")
        devices = [int(i) for i in args.gpus.split(',') if i.strip()]
        train_dataset, val_dataset, eval_metric = get_dali_dataset(args.dataset, devices, args)
        train_data, val_data = get_dali_dataloader(
            async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers,
            devices, ctx[0], args.horovod, args.seed)
    else:
        train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
        batch_size = (args.batch_size // hvd.size()) if args.horovod else args.batch_size
        train_data, val_data = get_dataloader(
            async_net, train_dataset, val_dataset, args.data_shape, batch_size, args.num_workers, ctx[0])



    # training
    train(net, train_data, val_data, eval_metric, ctx, args)