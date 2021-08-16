import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2, viz
cv2 = try_import_cv2()
import mxnet as mx

def plot_current_money(np_image,cid,score,lines):
    tongtien = 0
    money = lines[1]
    for cls,sco in zip(cid,score):
      #print(cls.asscalar())
      if sco.asscalar()>=0.4:
        try:
          tongtien+=int(money[int(cls.asscalar())])
        except:
          pass
        
    s = 'Tien: ' + str(tongtien)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1#int(max(im.shape[0],im.shape[1])/1024)+1
    org=(10*fontscale,50*fontscale)
    thickness = 2*fontscale
    color = (0, 255, 0)
    cv2.putText(np_image,s,org,font,fontscale,color,thickness)

def Detect_video(s,d,net,ctx,lines):
  cap= cv2.VideoCapture(s)
  fps = cap.get(cv2.CAP_PROP_FPS)
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  vid_writer = None
  tongtg = 0
  solan = 0
  while (cap.isOpened()):
    ret, frame = cap.read()
    #print(type(frame))
    if ret == False :#or time.time()-t1 > 10:
        break
    
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, 480)
    new_rgb_nd = rgb_nd.copyto(ctx[0])
    t1 = time.time()
    cid, score, bbox = net(new_rgb_nd)
    print(f'Done. ({time.time() - t1:.3f}s)')
    tongtg+=time.time() - t1
    solan+=1
    np_image = viz.cv_plot_bbox(frame, bbox[0], score[0], cid[0],thresh=0.4, class_names=classes)
    plot_current_money(np_image,cid[0],score[0],lines)
    if not isinstance(vid_writer, cv2.VideoWriter):
      vid_writer = cv2.VideoWriter(d, cv2.VideoWriter_fourcc(*'mp4v'), fps, (np_image.shape[1], np_image.shape[0]))
    
    vid_writer.write(np_image[...,::-1])
  vid_writer.release()
  cap.release()
  cv2.destroyAllWindows()
  print(tongtg/solan)

s = '/content/video/VID20210810151239.mp4' # Video muốn detect
d = '/content/video_ssd/VID20210810151239.mp4' # Đường dẫn lưu video đã detect
params = '/content/drive/MyDrive/GLUONCV/custom_ssd_512_resnet50_v1_coco_0039_0.9872.params' # File weights
net_name = 'ssd_512_resnet50_v1_coco'
csv_money_path = 'Classes.csv' # Đường dẫn tới file giá tiền

f_money = open(csv_money_path,'r')
lines = f_money.read().split('\n')
if lines[-1] == '':
    lines = lines[:-1]
for i in range(len(lines)):
    lines[i] = lines[i].split(',')
# Load the model
try:
  a = mx.nd.zeros((1,), ctx=mx.gpu(0))
  ctx = [mx.gpu(0)]
except:
  ctx = [mx.cpu()]

#ctx = [mx.cpu()]
net = gcv.model_zoo.get_model(net_name, pretrained=True)
# Compile the model for faster speed
net.hybridize()

# Tên các class theo đúng thứ tự như khi train
classes = ['coloscare_hong', 'coloscare_vang', 'sumi_orange', 'sumi_cherry', 'coloscare_xanh', 'smarta_grow', 'tobokki_vang', 'keo_kimiko', 'vansua_nestle', 'vansua_burine', 'rongbien_bbq', 'poro_do', 'poro_xanhla', 'kun_tim_suabich', 'kun_cam_suabich', 'poro_tim', 'poro_xanhdatroi', 'hipp_buckwheat', 'tobokki_do', 'kun_hong_suabich', 'kun_do_suabich', 'kun_xanhla_suabich', 'aptamil', 'hanie_kid', 'metacare_blue', 'coloscare_hong1', 'ostelin_kids', 'hd_kids', 'nutricare_bone', 'kazu_hong', 'leankid_100_tim', 'nestle_nan_xanhtroi', 'kazu_xanh', 'kazu_vang', 'botandam_ridielac_xanhtroi', 'suanon_goldilac', 'botandam_metacare_hong', 'botandam_metacare_xanh', 'bot_andam_metacare_vang', 'mamamy_daugoi', 'Lactacycl_bb', 'gohmo_suatam_nhat', 'gohmo_suatam_dam', 'cetaphil', 'pureen_suatam_trang', 'dnee_kid_do', 'dnee_kid_xanh', 'dr_smile', 'diepannhi', 'pureen_vang', 'ro_luoi', 'nuocmuoi', 'botandam_dielac_do', 'benokid_colos', 'vinamilk_suabich', 'fami_suabich', 'pentavite', 'nunest_hop', 'beno_kid', 'milktea_hong', 'matcha_vang', 'kazu_xanhla', 'khautrang_kid', 'elemis_xanh', 'ongtho_hop', 'suaoccho_nau', 'care_100_gold_trang', 'metacare_gold_xanhtroi', 'metacare_gold_xanhla', 'metacare_gold_hong', 'care_100_trang_xanh', 'goigiavi_xanhla', 'goigiavi_do', 'goigiavi_nau', 'meji_hong', 'meji_nau', 'metamom_tim', 'suaoccho_1hop', 'care100_xanhla_1loc', 'hero_cam_1loc', 'pediasure_xanhhong_1loc', 'growplus_do_1loc', 'leankid100_vang_1loc', 'smarta_vang_1loc', 'babyme_1loc', 'kazu_hong_1loc', 'enfagrow_1loc', 'kazu_xanh_1loc', 'Th_1loc', 'yoko_1loc', 'kun_do_1loc', 'grow_vang_1loc', 'care100_vang_1loc', 'optimum_gold_1loc', 'coloscare_trangxanh_1loc', 'vinamilk_itduong_1loc', 'grow_xanhtrang_1loc', 'metacare_xanhdam_1loc', 'alponte_1loc', 'dalatmilk_1loc', 'drluxia_dovang_1loc', 'khautrang_hopnho', 'botngucoc_nau', 'dau_chongmuoi', 'bocuoi_tron', 'bocuoi_bich', 'cheese_dessert_hong', 'cheese_dessert_xanh', 'goi_mau_vang', 'chocopie_hop', 'nuocyen_1loc', 'poro_bichxanh', 'poro_bichhong', 'chao_toyen_vang', 'chaotuoi_vang', 'chaotuoi_do', 'chaotuoi_xanhla', 'chaotuoi_hong', 'babyme_nest', 'vansua_bledina_hong', 'vansua_bledina_vang', 'suachua_hoff_xanhvang', 'suachua_hoff_vang', 'suachua_hoff_xanhtroi', 'suachua_hoff_do', 'khanuot_embe_do', 'khanuot_embe_xanhtroi', 'khanuot_embe_xanhla', 'khanuot_embe_vang', 'khanuot_trangsocxanh', 'tinhdau_xanhla', 'tinhdau_trang', 'ruatay_lifeboy', 'ruatay_24_vang', 'ruatay_suzu_xanh', 'ruatay_suzu_hong', 'hamsua_hong', 'hamsua_xanh', 'ensure_1loc', 'breastmilk_hong', 'binhsua_hongdo', 'breast_milk_xanh', 'breast_milk_trang_xanh', 'phan_pureen_trang', 'ta_popolin_trang', 'super_unidry_vangm28', 'bobby_40+4', 'huggies_dry_L', 'bobby_+6', 'bobby_extrasofting_L', 'binhsua_baby_xanhtroi', 'binhsua_baby_trangcao', 'binhsua_baby_hong', 'binhsua_baby_xanhla', 'binhsua_baby_vang', 'ho_trangxanh_hop', 'DHA_hongvang', 'dauphong_chai', 'banh_hellokitty', 'dauoliu_vangnhat', 'kiddy_chaiden', 'mam_vio_cam', 'mam_vio_xanhnhat', 'dau_oliu_dintel', 'dau_oliu_chailun', 'babybites_vang', 'babybites_do', 'babybites_xanh', 'tambong_xanhla', 'susu_nau_1loc', 'susu_cam_1loc', 'kemdanhrang_vang', 'kemdanhrang_do', 'susu_xanhla_1loc', 'tambong_xanhtroi', 'tatawa_cappucino_do', 'chao_wakodo_hong', 'keo_morinaga_do', 'chao_wakodo_xanhduong', 'nuoinhat_hong_60k', 'bunnhat_cam_60k', 'banhgao_treem_xanhla', 'banhgao_treem_xanhduong', 'dau_ajinomoto', 'binhsua_pur_xanhduong', 'binhsua_babiboo_cam', 'binhsua_latan_do', 'binhsua_ucare_cam', 'botdanhrang_pur', 'numvu_pur', 'khanvai_hientrang', 'khanvai_babyquy_nho', 'num_latan_tim', 'num_latan_xanhduong', 'num_latan_xanhla', 'num_pur_xanhduong', 'khangiay_emos_tim', 'vinamilk_itduong_hopto', 'asutralian100_hopto', 'giat_dnee_xanhduong', 'giat_fineline_tim', 'giat_dnee_xanhduong2', 'giat_fineline_hong', 'giat_dnee_xanhla', 'lausan_lovic_hong', 'bovesinh_babiboo_cam', 'origold_xanhduong', 'origold_tim', 'origold_xanhla', 'origold_hon']
#classes = ['bocuoi_bich']
for i in range(len(classes)):
  classes[i] = classes[i].lower()
net.reset_class(classes)
print('Loading parameters...')
net.load_parameters(params)
net.collect_params().reset_ctx(ctx[0])
Detect_video(s,d,net,ctx,lines)
