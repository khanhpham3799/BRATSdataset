import tarfile
import os
import json
from pathlib import Path
import random
import nibabel
import numpy as np 
import matplotlib.pyplot as plt

def main():
    #extract = True for extracting tar file 1st time
    org_dir = "/media/khanhpham/새 볼륨/BRats2021/"
    tar_extract_dir = org_dir + "BRATS"
    save_npz_dir = org_dir + "brats2021_64x64"
    if not os.path.exists(tar_extract_dir):
        os.mkdir(tar_extract_dir)
    if not os.path.exists(save_npz_dir):
        os.mkdir(save_npz_dir)
    extract = False
    if extract:
        filename = ["archive/BraTS2021_Training_Data.tar","archive/BraTS2021_00495.tar","archive/BraTS2021_00621.tar"]
        for name in filename:
            file_dir = org_dir + name
            zip_file = tarfile.open(file_dir)
            if name == "archive/BraTS2021_Training_Data.tar":
                extract_dir = tar_extract_dir 
            else:
                file_n = name.split("/")[-1]
                file_n = file_n.split(".")[0]
                extract_dir = tar_extract_dir + "/" + file_n 
            zip_file.extractall(extract_dir)
            zip_file.close()
    pt_dir = sorted(os.listdir(tar_extract_dir))[1:]
    print(pt_dir)

    ###divide train/test set
    json_dir = org_dir + "/list.json"
    if not os.path.exists(json_dir):
        #os.mkdir(json_dir)
        dict_list = {"train":[],"test":[]}
        for i in pt_dir:
            pt_no = i.split("_")[-1]
            pt_no = int(pt_no)
            if random.random() < 0.3:
                dict_list['test'].append(pt_no)
            else:
                dict_list['train'].append(pt_no)
        with open(json_dir, 'w') as f:
            json.dump(dict_list, f)
        train_list = dict_list['train']
        test_list = dict_list['test']
    else:
        f = open(json_dir)
        data =json.load(f)
        train_list = data['train']
        test_list = data['test']
    train_dir = save_npz_dir + "/npy_train"
    test_dir = save_npz_dir + "/npy_test"
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    ###

    ###extract train###
    for pt in train_list:
        pt_no = int(pt)
        pt_dir = train_dir + "/patient_" + str(f"{pt_no:05d}")
        if not os.path.exists(pt_dir):
            os.mkdir(pt_dir)
        load_dir = tar_extract_dir + str(f"/BraTS2021_{pt_no:05d}")

        flair_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_flair.nii.gz")
        flair = nibabel.load(flair_path).get_fdata() #shape (240,240,155)

        t1_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t1.nii.gz")
        t1 = nibabel.load(t1_path).get_fdata()

        t1ce_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t1ce.nii.gz")
        t1ce = nibabel.load(t1ce_path).get_fdata()

        t2_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t2.nii.gz")
        t2 = nibabel.load(t2_path).get_fdata()

        seg_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_seg.nii.gz")
        seg = nibabel.load(seg_path).get_fdata()

        for slice_no in range(flair.shape[2]):
            img = np.ones((4,flair.shape[0],flair.shape[1]))
            seg_img = np.ones((1,flair.shape[0],flair.shape[1]))
            img[0] = flair[:,:,slice_no]
            img[1] = t1[:,:,slice_no]     
            img[2] = t1ce[:,:,slice_no]     
            img[3] = t2[:,:,slice_no]
            seg_img[0] = seg[:,:,slice_no]
            slice_dir = pt_dir + str(f"/slice_{slice_no:03d}.npz")
            np.savez(slice_dir, x=img, y=seg_img)
        print("saved img:",slice_dir)
    ###

    ###extract test###
    for pt in test_list:
        pt_no = int(pt)
        pt_dir = test_dir + "/patient_" + str(f"{pt_no:05d}")
        if not os.path.exists(pt_dir):
            os.mkdir(pt_dir)
        load_dir = tar_extract_dir + str(f"/BraTS2021_{pt_no:05d}")

        flair_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_flair.nii.gz")
        flair = nibabel.load(flair_path).get_fdata() #shape (240,240,155)

        t1_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t1.nii.gz")
        t1 = nibabel.load(t1_path).get_fdata()

        t1ce_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t1ce.nii.gz")
        t1ce = nibabel.load(t1ce_path).get_fdata()

        t2_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_t2.nii.gz")
        t2 = nibabel.load(t2_path).get_fdata()

        seg_path = load_dir + str(f"/BraTS2021_{pt_no:05d}_seg.nii.gz")
        seg = nibabel.load(seg_path).get_fdata()

        for slice_no in range(flair.shape[2]):
            img = np.ones((4,flair.shape[0],flair.shape[1]))
            seg_img = np.ones((1,flair.shape[0],flair.shape[1]))
            img[0] = flair[:,:,slice_no]
            img[1] = t1[:,:,slice_no]     
            img[2] = t1ce[:,:,slice_no]     
            img[3] = t2[:,:,slice_no]
            seg_img[0] = seg[:,:,slice_no]
            slice_dir = pt_dir + str(f"/slice_{slice_no:03d}.npz")
            np.savez(slice_dir, x=img, y=seg_img)
        print("saved img:",slice_dir)
    ###
if __name__ == "__main__":
    main()
    '''
    test = nibabel.load("/home/khanhpham/khanhpham/data/BRats2021/BRATS/BraTS2021_00002/BraTS2021_00002_seg.nii.gz")
    test = test.get_fdata()
    print(test)
    print(test.shape)
    '''
    '''
    img_org = np.load("/home/khanhpham/khanhpham/data/BRats2021/brats2021_64x64/npy_train/patient_00009/slice_065.npz")
    img = img_org['x']
    fig = plt.figure(figsize=(64.,64.))
    row,col = 1,img.shape[0]
    for l in range(img.shape[0]):
        fig.add_subplot(row, col, l+1)
        x = img[l,...] 
        plt.imshow((x),cmap = 'gray')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        plt.axis("off")
    plt.savefig("/home/khanhpham/khanhpham/data/org.png")
    plt.clf()
    plt.close()
    '''