import datetime
import datetime
import os.path
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

from PIL import Image

import DCGAN
import imgtransform as imgt

dir_in = 'out/result'
dir_out = 'mosaic'
MOSAIC_NR = 5000
CELL_NR_MIN = 1
CELL_RANGE = 500
size = (512, 512) #size = (1936, 1456)


def createMosaic(id, count, dirpath):
    if not os.path.exists(dirpath + '/img'):
        os.mkdir(dirpath + '/img')
    if not os.path.exists(dirpath + '/dot'):
        os.mkdir(dirpath + '/dot')
    print(f'[{id:05d}] Creating mosaic with {count} cells')
    imgcnt = dirpath.split('_')[1]
    bg_mosaic = Image.new('RGB', size=size)
    bg_dot = Image.new('RGB', size=size)
    cellcnt = 1
    for cellcnt in range(count):
        img = getNewImage(1)[0]
        while not imgt.checkImage(img):
            img = getNewImage(1)[0]
        img = randomizer(img)
        img = imgt.convertAlpha(img)
        location_dot = (random.randint(0, size[0]), random.randint(0, size[1]))
        location_img = (location_dot[0] - int(img.size[0] / 2), location_dot[1] - int(img.size[1] / 2))
        bg_mosaic.paste(img, location_img, img)
        bg_dot.paste(imgt.dotAnnot(), location_dot)
        # print(f'\rGenerating cells: {cellcnt}/{count}', end=' ')
    bg_mosaic = imgt.imgBlur(bg_mosaic)
    bg_mosaic.save(f'{dirpath}/img/{imgcnt}_{id:05d}_{cellcnt+1:05d}_img.png')
    bg_dot.save((f'{dirpath}/dot/{imgcnt}_{id:05d}_{cellcnt+1:05d}_dot.png'))


def cellGeneration():
    img = getNewImage(1)[0]
    while not imgt.checkImage(img):
        img = getNewImage(1)[0]
    img = randomizer(img)
    img = imgt.convertAlpha(img)
    location_dot = (random.randint(0, size[0]), random.randint(0, size[1]))
    location_img = (location_dot[0] - int(img.size[0] / 2), location_dot[1] - int(img.size[1] / 2))
    return img, location_img, location_dot


def createMosaicFolder():
    ready = False
    foldercnt = 0
    dirpath = ""
    while not ready:
        nextpath = f'{dir_out}/{dir_out}_{foldercnt:03d}'
        if os.path.exists(nextpath):
            foldercnt += 1
        else:
            os.mkdir(nextpath)
            ready = True
            dirpath = nextpath
    return dirpath


def randomizer(img):
    img = imgt.imgResize(img, (random.randint(8, 12)) / 10, (random.randint(8, 12)) / 10)
    img = imgt.imgRotate(img, rotate=random.randint(0, 359))
    img = imgt.imgBrightness(img, random.randint(100, 140) / 100)
    img = imgt.imgAlpha(img, random.randint(80, 100) / 100)
    img = imgt.imgBlur(img, random.randint(0, 20))
    return img


def getNewImage(count, path=''):
    if path != '':
        img = (imgt.openDir(path))
    else:
        img = DCGAN.useModels(count)
    return img


def main():
    dir_path = createMosaicFolder()
    start = datetime.datetime.now()
    with open(f'{dir_path}/label.csv', "a") as myfile:
        myfile.write(f"id,count\n")
    futures = []
    with ProcessPoolExecutor() as pool:
        for x in range(int(MOSAIC_NR)):
            y = random.randint(1, CELL_RANGE) + CELL_NR_MIN
            futures.append(pool.submit(createMosaic, x+1, y, dir_path))
            time.sleep(1)
            with open(f'{dir_path}/label.csv', "a") as myfile:
                myfile.write(f"{x},{y}\n")
    imgt.createNpy(dir_path)
    end = datetime.datetime.now()
    print(f'\r{start} - {end}')
    print(f'{end - start}')


if __name__ == '__main__':
    freeze_support()
    main()
