import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

import shutil

# 初始化NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)

nusc_corcer_case = { "nuscenes_002733.jpg": "b8f8a7e278f04035abf83b57fde78fb8",
                     "nuscenes_002734.jpg": "d233a22c8c4546e5a5dd38c3d7452bce", 
                     "nuscenes_002950.jpg": "6b10154e2b3146ab85b43f3cfe053d6d", 
                     "nuscenes_002951.jpg": "1259283ec9f549d4b26c98ee76c654a6", 
                     "nuscenes_002952.jpg": "38bde80644fd47f4b08d8de873a8be1b", 
                     "nuscenes_002957.jpg": "1ecb668fec5d4f27bdcd6832ac687dce", 
                     "nuscenes_002959.jpg": "f9e3162f45db4bf5bc7f9360a831d2e4", 
                     "nuscenes_003640.jpg": "02fa71fc16d449c382c24b1b7deef3ba", 
                     "nuscenes_003641.jpg": "f4ee84975f4448c49532330e9efd9749", 
                     "nuscenes_003643.jpg": "8855fe65f4964e3e857cc84222b54d66", 
                     "nuscenes_003645.jpg": "9f4eea927ef549e19872cf4afd8990f5", 
                     "nuscenes_004128.jpg": "c158bbb8d938444a9feeb3752cc7607f", 
                     "nuscenes_004370.jpg": "a0ec29607e5644a4abfaef970542e123", 
                     "nuscenes_007731.jpg": "66e1216263b449dca6c7f0972bc82e0d", 
                     "nuscenes_007732.jpg": "7646a32158e1493db4eb380d682b261e", 
                     "nuscenes_007733.jpg": "8ee182e859f342cbb038864fac9ea87c", 
                     "nuscenes_007734.jpg": "c72ff6d2679f4f63b5bf53157d6af051", 
                     "nuscenes_007735.jpg": "c9f1dedf2573435b9aeff352d37f65a4", 
                     "nuscenes_007768.jpg": "7d82aba5838443f3a79bb37a4eeb48f9", 
                     "nuscenes_009604.jpg": "0a899fddb0794dbd9f9e1428d45d7446", 
                     "nuscenes_010820.jpg": "6780c0c8790a4beab54674a802577e26", 
                     "nuscenes_010824.jpg": "dbffc3f8df2f4bf69a1496e0275d63ed", 
                     "nuscenes_010825.jpg": "66a32c220006434fb0c50aa6bf5045b2", 
                     "nuscenes_010826.jpg": "2cbf41b797ee47538a1d59361ecbd6fe", 
                     "nuscenes_011001.jpg": "170f60c1fe9a46fa99d818ebe86282a1", 
                     "nuscenes_011715.jpg": "009cd1fd3c2048ebb6697832305b04eb", 
                     "nuscenes_012678.jpg": "d8f6a1cb324f46768d4bff2ec2b9b2f2", 
                     "nuscenes_012791.jpg": "994c260549a747be97c4b337c2e06873", 
                     "nuscenes_012801.jpg": "4d184d1e16c14a08af8f5cab36dca586", 
                     "nuscenes_012866.jpg": "a3b069be8cff4f6e9e3155a0bcd4dae1", 
                     "nuscenes_012882.jpg": "67b22c6b82b5403dae50d7e9242d375c", 
                     "nuscenes_012898.jpg": "bff7446f68ac4297a8fa4b0bfb0c9308", 
                     "nuscenes_015272.jpg": "7db4365073c04d949fbf858a635daf72", 
                     "nuscenes_015713.jpg": "3f290bfc6ffe475f9774784c0d0e4538", 
                     "nuscenes_016240.jpg": "eac471efcfa0495685b9669f91548bce", 
                     "nuscenes_016241.jpg": "f86bfac9b34942faacb05ac46eb39311", 
                     "nuscenes_016243.jpg": "44523fb5486b41c78e7b2233fad7188e", 
                     "nuscenes_016440.jpg": "1a1dc7ef4955481ba3755fbaa39a53d8", 
                     "nuscenes_016441.jpg": "ae68429340df438aa95138c9ea6c8df1", 
                     "nuscenes_016460.jpg": "a3da8976af104ae7a48a186c610a229a", 
                     "nuscenes_016461.jpg": "6cd73bca1f384f968b06466eaf509383", 
                     "nuscenes_016486.jpg": "adcbf5d4a1fb41a08aadf1dea4da8d66", 
                     "nuscenes_016488.jpg": "ea43c86a960344f09d3b3ab2035aced8", 
                     "nuscenes_016491.jpg": "17dc38b84c1049cdb8891866948a25f0", 
                     "nuscenes_016493.jpg": "6e5f83862953468098943f4901e9551a", 
                     "nuscenes_017062.jpg": "821f64c5e5dc4d6f8eb1e2bdcaeda635", 
                     "nuscenes_018945.jpg": "49514071dcad40a0ae974d6a1f9db68a", 
                     "nuscenes_018946.jpg": "3d8c2624af7b4e27b0c780f87f5bc4af", 
                     "nuscenes_018948.jpg": "ab013716d21c4f39ac6446553a44cf37", 
                     "nuscenes_018949.jpg": "d7e7493e09ab414ebec72a327358ce84", 
                     "nuscenes_018950.jpg": "46d55dd1cebd411bb81d44c11265d659", 
                     "nuscenes_018951.jpg": "3519f9fb21374105865ae9eadc76ff3a", 
                     "nuscenes_018952.jpg": "d388750e0be44b26a000ef96728aab2e", 
                     "nuscenes_019349.jpg": "101a617fba0a4c7a91073b49b9b6752f", 
                     "nuscenes_019350.jpg": "178c06a23bbc4b0cb34a8708f48cf88c", 
                     "nuscenes_021213.jpg": "0ed34ecd25444039b6ef5826dc951c4e", 
                     "nuscenes_021214.jpg": "d709c01d993f408988697cf67a50e061", 
                     "nuscenes_021216.jpg": "63719ab6198546ad9b2d9085fa1b1310", 
                     "nuscenes_021217.jpg": "b8e21821a260483e87196b485394cd32", 
                     "nuscenes_021218.jpg": "f709cc87399044be9937b641caa15d30", 
                     "nuscenes_021348.jpg": "06e25402b8b04323a04644c147bea8bb", 
                     "nuscenes_021349.jpg": "0bcc3b17ac134d6e8a15de8380228c41", 
                     "nuscenes_021351.jpg": "ffd5b4708e5f4f2ea476dbde3793d48e", 
                     "nuscenes_021354.jpg": "e51849eaf8ad41bdae5fccabf1222c03", 
                     "nuscenes_021355.jpg": "7b3af3a4394c4270b11dc303bbe967e2", 
                     "nuscenes_021357.jpg": "74bc3f39676a4a19a0442682ff82f01b", 
                     "nuscenes_021359.jpg": "011b8e758330400dbdf16d55c8a3571a", 
                     "nuscenes_021887.jpg": "2f0ad04cba5547cfb6a582b9bccd53e3", 
                     "nuscenes_021888.jpg": "8c5e975bb5d94261b4edc11001f17b19", 
                     "nuscenes_021889.jpg": "7789588e877d4e28a5061b195eb676a6", 
                     "nuscenes_021890.jpg": "4a3f7ebe6a98411c9c60c00e8e0b064e", 
                     "nuscenes_021891.jpg": "c48039e45a4c4a91a681ec23f599b9a6",
                     "nuscenes_021892.jpg": "e776e7a70fe5402db59db60ce9642d6d", 
                     "nuscenes_023034.jpg": "5557ac1bef274c8b940be57ffac9afd0", 
                     "nuscenes_023672.jpg": "191a1459e84d499f96cb6ed0cabc0f7b", 
                     "nuscenes_025586.jpg": "88d3d469b048411baad87fee71910676", 
                     "nuscenes_025610.jpg": "98021fc238e04523891345bdc307a750", 
                     "nuscenes_025611.jpg": "79c4b248e94f449f9d01447096261d26", 
                     "nuscenes_025627.jpg": "d9d7315b65054f269268c1aa64eb6289", 
                     "nuscenes_025629.jpg": "0ccae58ca1a44dc3ac2be3fd48389dd7", 
                     "nuscenes_027283.jpg": "195362bd105443c8a5b35d9fa84795c7", 
                     "nuscenes_027903.jpg": "86f869ba553c4bc99233b8a2986f7679", 
                     "nuscenes_029236.jpg": "6d2793d97afe4a06b9081dea4c954050", 
                     "nuscenes_029771.jpg": "1cb5b70b0eb64e2ebec2c9608583a3b3", 
                     "nuscenes_029816.jpg": "fcaaa2f2f3bb46e39f04cdd16a1a53d3", 
                     "nuscenes_030613.jpg": "45399c614d154029a3898ba78a1d0e0f", 
                     "nuscenes_033205.jpg": "238e18bd4f0f49c489ecad19b47c3a6a", 
                     "nuscenes_033402.jpg": "1a41ba0751d5497ebd32df7c86950671", 
                     "nuscenes_033933.jpg": "0b829a1f8f024f1da39208c0f7effb3c",}

# 使用sample token调用函数
idx = 0
dst_dir = 'data/nuscenes_corner_case'

for k,v in nusc_corcer_case.items():
    sample_token = v
    sensor = 'CAM_FRONT'

    my_sample = nusc.get('sample', sample_token)
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    filename = 'data/nuscenes/' + cam_front_data['filename']
    new_filename = f'{idx}_raw.png'
    dst_path = os.path.join(dst_dir, new_filename)
    shutil.copy(filename, dst_path)
    print(f'copy {idx} gt')
    
    nusc.render_sample_data(cam_front_data['token'], out_path=f'./data/nuscenes_corner_case/{idx}_gt.png')
    print(f'save {idx} gt')
    idx +=1