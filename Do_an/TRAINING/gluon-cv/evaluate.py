import os
import sys
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
from os.path import join
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.accuracy import Accuracy

from mxnet.contrib import amp
from gluoncv.data import VOCDetection

def get_yolo_predicted_result(img_path, txt_path, width, height):
  '''Get yolo predicted result (txt annoation)'''
  def read_yolo_txt_file_with_scores(path,width,height):
    f = open(path,'r')
    results = f.read().split('\n')
    if results[-1] == '':
      results = results[:-1]

    bboxes = []
    classes = []
    scores = []
    results.reverse()
    for result in results:
      l = list(map(float,result.split()))
      X1 = round(l[1]*width-l[3]*width/2)
      Y1 = round(l[2]*height-l[4]*height/2)
      X2 = round(l[1]*width+l[3]*width/2)
      Y2 = round(l[2]*height+l[4]*height/2)
      C = int(l[0])
      if l[5]>=0.01:
        bboxes.append([X1,Y1,X2,Y2])
        classes.append([C])
        scores.append([l[5]])
      else:
        bboxes.append([-1,-1,-1,-1])
        classes.append([-1])
        scores.append([-1])
    return [bboxes, classes, scores]

  img = mx.img.imread(img_path)
  arr = read_yolo_txt_file_with_scores(txt_path, width, height) #640 360
  img = gcv.data.transforms.image.imresize(img, width, height)
  # box, cls. score
  return mx.nd.array([arr[0]]),mx.nd.array([arr[1]]),mx.nd.array([arr[2]])

def get_yolo_gt_result(img_path, txt_path, width, height):
  '''Get yolo ground-truth result (txt annoation)'''
  def read_yolo_txt_file(path,width,height):
    f = open(path,'r')
    results = f.read().split('\n')
    if results[-1] == '':
      results = results[:-1]

    bboxes = []
    classes = []
    for result in results:
      l = list(map(float,result.split()))
      X1 = round(l[1]*width-l[3]*width/2)
      Y1 = round(l[2]*height-l[4]*height/2)
      X2 = round(l[1]*width+l[3]*width/2)
      Y2 = round(l[2]*height+l[4]*height/2)
      C = int(l[0])
      bboxes.append([X1,Y1,X2,Y2])
      classes.append([C])
    return [bboxes, classes]

  img = mx.img.imread(img_path)
  arr = read_yolo_txt_file(txt_path, width, height) #640 360
  img = gcv.data.transforms.image.imresize(img, width, height)
  # box, cls. score
  return mx.nd.array([arr[0]]),mx.nd.array([arr[1]])

def get_ssd_predicted_result(net,ctx,img_path,width):
  '''Get mxnet ssd predicted result'''
  x, image = gcv.data.transforms.presets.ssd.load_test(img_path,width)
  new_x = x.as_in_context(ctx[0])
  cid, score, bbox = net(new_x)
  return bbox, cid, score
if __name__ == "__main__":
  #classes
  classes = ['coloscare_hong', 'coloscare_vang', 'sumi_orange', 'sumi_cherry', 'coloscare_xanh', 'smarta_grow', 'tobokki_vang', 'keo_kimiko', 'vansua_nestle', 'vansua_burine', 'rongbien_bbq', 'poro_do', 'poro_xanhla', 'kun_tim_suabich', 'kun_cam_suabich', 'poro_tim', 'poro_xanhdatroi', 'hipp_buckwheat', 'tobokki_do', 'kun_hong_suabich', 'kun_do_suabich', 'kun_xanhla_suabich', 'aptamil', 'hanie_kid', 'metacare_blue', 'coloscare_hong1', 'ostelin_kids', 'hd_kids', 'nutricare_bone', 'kazu_hong', 'leankid_100_tim', 'nestle_nan_xanhtroi', 'kazu_xanh', 'kazu_vang', 'botandam_ridielac_xanhtroi', 'suanon_goldilac', 'botandam_metacare_hong', 'botandam_metacare_xanh', 'bot_andam_metacare_vang', 'mamamy_daugoi', 'Lactacycl_bb', 'gohmo_suatam_nhat', 'gohmo_suatam_dam', 'cetaphil', 'pureen_suatam_trang', 'dnee_kid_do', 'dnee_kid_xanh', 'dr_smile', 'diepannhi', 'pureen_vang', 'ro_luoi', 'nuocmuoi', 'botandam_dielac_do', 'benokid_colos', 'vinamilk_suabich', 'fami_suabich', 'pentavite', 'nunest_hop', 'beno_kid', 'milktea_hong', 'matcha_vang', 'kazu_xanhla', 'khautrang_kid', 'elemis_xanh', 'ongtho_hop', 'suaoccho_nau', 'care_100_gold_trang', 'metacare_gold_xanhtroi', 'metacare_gold_xanhla', 'metacare_gold_hong', 'care_100_trang_xanh', 'goigiavi_xanhla', 'goigiavi_do', 'goigiavi_nau', 'meji_hong', 'meji_nau', 'metamom_tim', 'suaoccho_1hop', 'care100_xanhla_1loc', 'hero_cam_1loc', 'pediasure_xanhhong_1loc', 'growplus_do_1loc', 'leankid100_vang_1loc', 'smarta_vang_1loc', 'babyme_1loc', 'kazu_hong_1loc', 'enfagrow_1loc', 'kazu_xanh_1loc', 'Th_1loc', 'yoko_1loc', 'kun_do_1loc', 'grow_vang_1loc', 'care100_vang_1loc', 'optimum_gold_1loc', 'coloscare_trangxanh_1loc', 'vinamilk_itduong_1loc', 'grow_xanhtrang_1loc', 'metacare_xanhdam_1loc', 'alponte_1loc', 'dalatmilk_1loc', 'drluxia_dovang_1loc', 'khautrang_hopnho', 'botngucoc_nau', 'dau_chongmuoi', 'bocuoi_tron', 'bocuoi_bich', 'cheese_dessert_hong', 'cheese_dessert_xanh', 'goi_mau_vang', 'chocopie_hop', 'nuocyen_1loc', 'poro_bichxanh', 'poro_bichhong', 'chao_toyen_vang', 'chaotuoi_vang', 'chaotuoi_do', 'chaotuoi_xanhla', 'chaotuoi_hong', 'babyme_nest', 'vansua_bledina_hong', 'vansua_bledina_vang', 'suachua_hoff_xanhvang', 'suachua_hoff_vang', 'suachua_hoff_xanhtroi', 'suachua_hoff_do', 'khanuot_embe_do', 'khanuot_embe_xanhtroi', 'khanuot_embe_xanhla', 'khanuot_embe_vang', 'khanuot_trangsocxanh', 'tinhdau_xanhla', 'tinhdau_trang', 'ruatay_lifeboy', 'ruatay_24_vang', 'ruatay_suzu_xanh', 'ruatay_suzu_hong', 'hamsua_hong', 'hamsua_xanh', 'ensure_1loc', 'breastmilk_hong', 'binhsua_hongdo', 'breast_milk_xanh', 'breast_milk_trang_xanh', 'phan_pureen_trang', 'ta_popolin_trang', 'super_unidry_vangm28', 'bobby_40+4', 'huggies_dry_L', 'bobby_+6', 'bobby_extrasofting_L', 'binhsua_baby_xanhtroi', 'binhsua_baby_trangcao', 'binhsua_baby_hong', 'binhsua_baby_xanhla', 'binhsua_baby_vang', 'ho_trangxanh_hop', 'DHA_hongvang', 'dauphong_chai', 'banh_hellokitty', 'dauoliu_vangnhat', 'kiddy_chaiden', 'mam_vio_cam', 'mam_vio_xanhnhat', 'dau_oliu_dintel', 'dau_oliu_chailun', 'babybites_vang', 'babybites_do', 'babybites_xanh', 'tambong_xanhla', 'susu_nau_1loc', 'susu_cam_1loc', 'kemdanhrang_vang', 'kemdanhrang_do', 'susu_xanhla_1loc', 'tambong_xanhtroi', 'tatawa_cappucino_do', 'chao_wakodo_hong', 'keo_morinaga_do', 'chao_wakodo_xanhduong', 'nuoinhat_hong_60k', 'bunnhat_cam_60k', 'banhgao_treem_xanhla', 'banhgao_treem_xanhduong', 'dau_ajinomoto', 'binhsua_pur_xanhduong', 'binhsua_babiboo_cam', 'binhsua_latan_do', 'binhsua_ucare_cam', 'botdanhrang_pur', 'numvu_pur', 'khanvai_hientrang', 'khanvai_babyquy_nho', 'num_latan_tim', 'num_latan_xanhduong', 'num_latan_xanhla', 'num_pur_xanhduong', 'khangiay_emos_tim', 'vinamilk_itduong_hopto', 'asutralian100_hopto', 'giat_dnee_xanhduong', 'giat_fineline_tim', 'giat_dnee_xanhduong2', 'giat_fineline_hong', 'giat_dnee_xanhla', 'lausan_lovic_hong', 'bovesinh_babiboo_cam', 'origold_xanhduong', 'origold_tim', 'origold_xanhla', 'origold_hon']
  
  #VOC 2007 mAP
  eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
  eval_metric.reset()
  
  #Tên model : ['yolov5','ssd_512_resnet50_v1']
  model = 'yolov5'
  
  # File hình tập test
  img_path = '/content/test/images'
  
  #Đường dẫn tới folder chứa labels của YOLOv5 đã detect từ tập test (confidence score 0 và số  boungding box là 100)
  pred_path = '/content/drive/MyDrive/TRAINING_ML/yolov5/runs/detect/results/labels/'
  
  #Đường dẫn tới labels đúng của tập test
  gt_path = '/content/test/labels/'

  #Đường dẫn tới file weights của ssd
  params = '/content/drive/MyDrive/GLUONCV_2/custom_ssd_512_resnet50_v1_coco_0066_0.9867.params'

  if model == 'yolov5':
    images = glob.glob(join(img_path,'*.*'))
    for image in images:
      name = image.split('/')
      if name[-1] == '':
        name=name[-2][:-4]
      else:
        name=name[-1][:-4]
      pred_anno = join(pred_path,name + '.txt')
      gt_anno = join(gt_path,name + '.txt')
      gt_bboxes, gt_ids = get_yolo_gt_result(image, gt_anno, 640, 360)
      det_bboxes, det_ids, det_scores = get_yolo_predicted_result(image,pred_anno,640,360)
      eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
  elif model == 'ssd_512_resnet50_v1':
    try:
      a = mx.nd.zeros((1,), ctx=mx.gpu(0))
      ctx = [mx.gpu(0)]
    except:
      ctx = [mx.cpu()]
    
    images = glob.glob(join(img_path,'*.*'))
    net = get_model(model + '_coco',pretrained=True)
    net.reset_class(classes)
    net.load_parameters(params)
    net.collect_params().reset_ctx(ctx[0])
    for image in images:
      name = image.split('/')
      if name[-1] == '':
        name=name[-2][:-4]
      else:
        name=name[-1][:-4]
      gt_anno = join(gt_path,name + '.txt')
      gt_bboxes, gt_ids = get_yolo_gt_result(image, gt_anno, 640, 360)
      det_bboxes, det_ids, det_scores = get_ssd_predicted_result(net,ctx,image,360)
      eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)

  map_name, mean_ap = eval_metric.get()

  recall, precision = eval_metric._recall_prec()
  val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
  print(val_msg)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logger.info('Evaluation: \n{}'.format(val_msg))
