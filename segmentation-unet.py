import os
import glob
import numpy as np

#Verilerimizi glob modülü ile alıp verisetimizi train ve test olarak ayırıyoruz.
def load_data(path):
    
  train_x = sorted(glob.glob(os.path.join(path,"training","images","*.tif")))
  train_y = sorted(glob.glob(os.path.join(path,"training","1st_manual","*.gif")))
  test_x =  sorted(glob.glob(os.path.join(path,"test","images","*.tif")))
  test_y =  sorted(glob.glob(os.path.join(path,"test","1st_manual","*.gif")))
  
  return (train_x,train_y),(test_x,test_y)

path="C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/Drive dataset"


(train_x, train_y), (test_x, test_y) = load_data(path)

from tqdm import tqdm
import imageio
import cv2

#Verilerimizi jpg haline dönüştürüp yeniden boyutlandırıyoruz (256,256,3) veData Augmentation yapıyoruz.
#Albumentations kütüphanesini install ediyoruz.(pip install -U albumentations)
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion

def augment_data(images,mask,save_path,augment=True):
  H = 256
  W = 256
  print(save_path)
  for idx, (x,y) in tqdm(enumerate(zip(images,mask)),total=len(images)):
    #Görüntünün sadece isim kısmını alıyoruz ilk kısımlarını atıyoruz.(21_training vb.)
    name = x.split("/")[- 1].split(".")[0]
    print(name)
    
    x = cv2.imread(x,cv2.IMREAD_COLOR)
    y = imageio.mimread(y)[0]
    #Data augmentation yöntemlerini uyguluyoruz.
    if augment == True:
         
        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x,mask=y)
        x1 = augmented['image']
        y1 = augmented['mask']
      
        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x,mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']
      
        aug = ElasticTransform(p=1.0,alpha=120,sigma=120*0.05)
        augmented = aug(image=x,mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']
      
        aug = GridDistortion(p=1.0)
        augmented = aug(image=x,mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']
        X = [x,x1,x2,x3,x4]
        Y = [y,y1,y2,y3,y4]
    
    else:  
        X = [x]
        Y = [y]
    index = 0
    #verilerimizi yeniden boyutlandırıp jpg haline dönüştürüp kaydediyoruz.(256,256,3)
    for i,m in zip(X,Y):
        i = cv2.resize(i,(W,H))
        m = cv2.resize(m,(W,H))
   
        if len(X) == 1:
          
          tmp_image_name = f"{name}.jpg"
          tmp_mask_name = f"{name}.jpg"
          
        else:
          
          tmp_image_name = f"{name}_{index}.jpg"
          tmp_mask_name = f"{name}_{index}.jpg"
          
        image_path = os.path.join(save_path,"images",tmp_image_name)
        mask_path = os.path.join(save_path,"mask",tmp_mask_name)
        print(image_path)
        print(mask_path)
        cv2.imwrite(image_path, i)
        cv2.imwrite(mask_path, m)
        index+=1

#augmenteation edilmiş veriler için yer oluşturuyoruz.
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data/train/image")
create_dir("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data/train/mask")
create_dir("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data/test/image")
create_dir("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data/test/mask")


augment_data(train_x,train_y,"C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data/train",True)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam

#Encoder blogumuz VGG-16 içinde daha önceden eğitilmiş kısımlar olacağı için o blok içindeki conv model tanımlandı.
def conv_block(inputs,num_filters):
    
  x = Conv2D(num_filters,3,padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_filters,3,padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  return x
#Decoder blogumuzu tanımladık.
def define_decoder(inputs,skip_layer,num_filters):
    
  init = RandomNormal(stddev=0.02)
  x = Conv2DTranspose(num_filters,(2,2),strides=(2,2),padding='same',kernel_initializer=init)(inputs)
  g = Concatenate()([x,skip_layer])
  g = conv_block(g,num_filters)
  
  return g
#VGG-16 modelimizi yüklüyoruz ve özetini alıyoruz.
vgg16 = VGG16(include_top=False,weights='imagenet')
vgg16.summary()

#VGG-16-U-Net Modelimizi kuruyoruz.
def vgg16_unet(input_shape):
  inputs = Input(shape=input_shape)
  vgg16 = VGG16(include_top=False,weights='imagenet',input_tensor=inputs)
  #Enoder layerlar için VGG-16'daki çıkış nöronlarını elde ediyoruz.(64,128,256,512)
  s1 = vgg16.get_layer('block1_conv2').output  
  s2 = vgg16.get_layer('block2_conv2').output  
  s3 = vgg16.get_layer('block3_conv3').output  
  s4 = vgg16.get_layer('block4_conv3').output  
  #burası VGG-16'nın dirsek kısmını oluşturuyor.
  b1 = vgg16.get_layer('block5_conv3').output 
  
  #Decoder kısmını oluşturuyoruz.
  d1 = define_decoder(b1,s4,512)
  d2 = define_decoder(d1,s3,256)
  d3 = define_decoder(d2,s2,128)
  d4 = define_decoder(d3,s1,64)
  
  outputs = Conv2D(1,1,padding='same',activation='sigmoid')(d4)
  model = Model(inputs,outputs)
  
  return model



# Dosyaları kaydetmek için yer oluşturuyoruz.
create_dir("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi")
#Hyperparameterlerimizi belirtiyoruz.
batch_size=2
lr = 1e-4
num_epochs = 500
model_path = os.path.join("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi","model.h5")
csv_path = os.path.join("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi","data.csv")

#Verilen path'e göre resimlerimizi yüklüyoruz. 
def load_path(path):
  X = sorted(glob(os.path.join(path,"images","*.jpg")))
  Y = sorted(glob(os.path.join(path,"mask","*.jpg")))
  return X,Y


from sklearn.utils import shuffle
#Resimlerimizi karıştırıyoruz.
def shuffling(x,y):
  x,y = shuffle(x,y)
  return x,y

#Yeni oluşturduğumuz veriler için path oluşturuyoruz.
dataset_path = "C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/new_data"

train_path = os.path.join(dataset_path,"train")
valid_path = os.path.join(dataset_path,"test")

#Verilerimizi yükleyip karıştırıyoruz.
train_x,train_y = load_path(train_path)
train_x,train_y = shuffling(train_x,train_y)
valid_x, valid_y = load_path(valid_path)

#Resimlerimizi normalize ediyoruz.(StandartScaler işlemi)
def read_image(path):
    
  path = path.decode()
  x = cv2.imread(path,cv2.IMREAD_COLOR)
  x = x/255.0
  x = x.astype(np.float32)
  
  return x

#Maskelerimizi normalize edip 3D tensor haline getiriyoruz.
def read_mask(path):
  path = path.decode()
  x = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
  x = x/255.0
  x = x.astype(np.float32)
  x = np.expand_dims(x,axis=-1) #(256,256,1)
  return x


#görüntüleri ve maskelerimizi işleyebilmek için parse ediyoruz.
def tf_parse(x,y):
    
  def _parse(x,y):
      
    x = read_image(x)
    y = read_mask(y)
    return x,y
  x,y = tf.numpy_function(_parse,[x,y],[tf.float32,tf.float32])
  x.set_shape([256,256,3])
  y.set_shape([256,256,1])
  
  return x,y


#Tensorflow Dataset API'yı kullanarak verilerimiz üzerinde parse,prefetch ve map işlemlerini uygulayıp batch'lere bölüyoruz.
def tf_dataset(X,Y,batch_size=2):
    
  dataset = tf.data.Dataset.from_tensor_slices((X,Y))
  dataset = dataset.map(tf_parse)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(4)
  
  return dataset


#Datasetlerimizi alıyoruz.
train_dataset = tf_dataset(train_x,train_y)
valid_dataset = tf_dataset(valid_x,valid_y)

#Eğitim ve Validasyon veri setlerimizi alıyoruz.
valid_x = np.array(valid_x)
valid_y = np.array(valid_y)

train_x = np.array(train_x)
train_y = np.array(train_y)

print(train_x.shape) #(75,)
print(train_y.shape) #(75,)

print(valid_x.shape) #(5,)
print(valid_y.shape) #(5,)


#eğitim step sayılarını oluşturuyoruz.
train_steps = len(train_x)//batch_size
test_steps = len(valid_x)//batch_size
if len(train_x) % batch_size != 0:
  train_steps +=1
if len(valid_x) % batch_size != 0:
  test_steps +=1
print(train_steps) #39
print(test_steps) #4

#Modelimizi değerlendirmek için iou metriğini tanımlıyoruz.
def iou(y_true,y_pred):
    
  def f(y_true,y_pred):
      
    intersection = (y_true*y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    x = x.astype(np.float32)
    
    return x

  return tf.numpy_function(f,[y_true,y_pred],tf.float32)

#Modelimizi değerlendirmek için dice_coef metriğini ve loss için dice_loss oluşturuyoruz.
smooth = 1e-15
def dice_coef(y_true,y_pred):
    
  y_true = tf.keras.layers.Flatten()(y_true)
  y_pred = tf.keras.layers.Flatten()(y_pred)
  intersection = tf.reduce_sum(y_true*y_pred)
  
  return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

def dice_loss(y_true,y_pred):
    
  return 1.0 - dice_coef(y_true,y_pred)

from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger



#kodlarımızı doğrudan çalıştırdık.
if __name__ == "__main__":

    #Modelimizi kaydediyoruz.
    model.save("C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/500epochs_model_vsl_loss.h5")


    

    #Elde ettiğimiz sonuçları,manuel segmentasyon görüntüsünü ve gerçek görüntüyü bir arada olucak şekilde kaydediyoruz.
    def save_results(ori_x,ori_y,y_pred,save_image_path):
    
      line = np.ones((256,10,3))*255
      ori_y = np.expand_dims(ori_y,axis=-1)
      ori_y = np.concatenate([ori_y,ori_y,ori_y],axis=-1)
      y_pred = np.expand_dims(y_pred,axis=-1)
      y_pred = np.concatenate([y_pred,y_pred,y_pred],axis=-1) * 255
      cat_images = np.concatenate([ori_x,line,ori_y,line,y_pred],axis=1)
      cv2.imwrite(save_image_path,cat_images)






    #Tahmin yapıp metric değerlerini hesaplıyoruz.
    def make_Predictions(mask_index,model):
    
      for x,y in tqdm(zip(test_x,test_y),total=len(test_x)):
        #Görüntünün ismini çıkarıyoruz.
        name = x.split ("\\") [- 1] .split (".") [0]
        #Görüntü ve maskeleri okuyoruz.
        ori_x,x = read_image(x) #(256,256,3)
        ori_y,y = read_mask(y)
        #Tahmin yapıyoruz.
        y_pred = model.predict(np.expand_dims(x,axis=0))[0] # (1,256,256,3)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = np.squeeze(y_pred,axis=-1)
 
        #Görüntüleri kaydediyoruz.
        save_image_path = f"C:/Users/ERKAN EROL/Desktop/Yapay Zeka Dersler/1.Sınıf Güz Dönemi/Yapay Sinir Ağı Uygulamaları/final ödevi/Sonuçlar/(name)_{index+mask_index}.png"
        save_results(ori_x,ori_y,y_pred,save_image_path)


#VGG-16 modelimizi tanımlıyoruz.
model = vgg16_unet((256,256,3))
#Modelimizi compile ediyoruz.
model.compile(loss=dice_loss,optimizer=Adam(lr),metrics=[dice_coef,iou,Recall(),Precision()])

#Callbacklerimizi tanımlıyoruz ve loss değerlerini kaydetmek için CSVLogger kuruyoruz
callbacks = [ModelCheckpoint(model_path,verbose=1,save_best_only=True),CSVLogger(csv_path)]

#Modelimizi 500 Epoch boyunca eğitiyoruz.
model.fit(train_dataset,epochs=500,validation_data=valid_dataset,steps_per_epoch=train_steps,validation_steps=test_steps)