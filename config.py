
class Config:
    data_path = 'data/faces'
    num_workers = 4
    image_size = 96
    max_epoch = 200
    ndf = 64            #判别器feature map数
    ngf = 64            #生成器feature map数