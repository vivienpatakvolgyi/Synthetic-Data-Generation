import os

import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Resampling


def imgRotate(img, path='', rotate=0):
    if rotate == 0:
        for x in range(4):
            img = img.rotate(rotate)
            rotate = rotate + 90
            if path != '':
                img.save(f'{path}_{rotate}.png')
            else:
                return img
    else:
        img = img.rotate(rotate)
        img = img.convert('RGB')
        if path != '':
            img.save(f'{path}_rotate{rotate}.png')
        else:
            return img


def imgFlip(img, path='', flip='V'):
    if flip == 'V':
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip == 'H':
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    else:
        if path != '':
            print('No axis given, flipping both ways and saving file: ' + path + '_(V/H).jpg')
            imgFlip(img, path, flip='V')
            imgFlip(img, path, flip='H')
        else:
            print('At least an axis or a save path needed')

    if path != '':
        img.save(f'{path}_flip{flip}.png')
    else:
        return img


def imgBlur(img, radius=2, path=''):
    img = img.filter(ImageFilter.BoxBlur(radius))
    if path != '':
        img.save(f'{path}_blur{radius}.png')
    else:
        return img


def imgResize(img, factorx, factory, path=''):
    img = img.resize((int(img.size[0] * factorx), int(img.size[1] * factory)), Resampling.NEAREST, )
    # img = img.convert('RGB')
    if path != '':
        img.save(path + f'_size{factorx}-{factory}.jpg')
    else:
        return img


def imgBrightness(img, factor, path=''):
    newData = []
    datas = img.getdata()
    for item in datas:
        r = min(int(item[0] * factor), 255)
        b = min(int(item[1] * factor), 255)
        g = min(int(item[2] * factor), 255)
        newData.append((r, b, g))
    img.putdata(newData)

    # img = img.convert('RGB')
    if path != '':
        img.save(path + f'_brightness{factor}.jpg')
    else:
        return img


def imgAlpha(img, factor, path=''):
    newData = []
    img = img.convert('RGBA')
    datas = img.getdata()
    for item in datas:
        alpha = min(int(item[3] * factor), 255)
        newData.append((item[0], item[1], item[2], alpha))
    img.putdata(newData)

    if path != '':
        img.save(path + f'_alpha{factor}.jpg')
    else:
        return img


def convertAlpha(img):
    newData = []
    datas = img.getdata()
    for item in datas:
        alpha = (item[0]*0.2162 + item[1]*0.7152 + item[2]*0.0722)
        if alpha > 255:
            alpha = 255
        newData.append((item[0], item[1], item[2], int(alpha)))
    img = Image.new('RGBA', size=img.size)
    img.putdata(newData)
    return img


def openImg(path):
    img = Image.open(path)
    img = img.convert('RGBA')
    return img


def openDir(dir_path):
    img_array = []
    for filename in os.listdir(dir_path):
        img_array.append(openImg(dir_path + '/' + filename))
    return img_array


def checkImage(img, min=80000, max=250000):
    datas = img.getdata()
    val = 0
    for item in datas:
        val += (item[0] + item[1] + item[2])
    if val <= min or val >= max:
        #print(f'Invalid picture! Sum color value: {val} ({min}-{max})')
        return False
    else:
        return True

def createNpy(path):
    idirs = os.listdir(path +'/img/')
    ddirs = os.listdir(path +'/dot/')
    idirs.sort()
    ddirs.sort()
    img_train = []
    dot_train = []
    if not os.path.exists(path+'/npy'):
        os.mkdir(path+'/npy')
    for item in idirs:
        im = Image.open(path +'/img/'+ item).convert("RGB")
        im = imgResize(im,0.25,0.25)
        im = np.array(im)
        img_train.append(im)
    for item in ddirs:
        im = Image.open(path +'/dot/'+ item).convert("RGB")
        im = np.array(im)
        dot_train.append(im)
    imgset = np.array(img_train)
    dotset = np.array(dot_train)
    np.save(path+"/npy/imgs.npy", imgset)
    np.save(path+"/npy/dots.npy", dotset)

def dotAnnot(size=1):
    dot = Image.new('RGB', size=(size,size))
    datas = dot.getdata()
    newdata=[]
    x=0
    for item in datas:
        item = (255, 0, 0)
        newdata.append(item)
        x += 1
    dot.putdata(newdata)
    return dot
# ----------------------------------------------------------------------------------------------------------------------

def dcgan_prep():
    images = openDir('lum_crop')
    for image in images:
        imgRotate(img=image, path='lum_crop_rot')
        imgFlip(img=image, path='lum_crop_rot')

