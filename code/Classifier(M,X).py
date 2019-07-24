from datetime import timedelta, datetime
import os
import shutil

#train
flare_train = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_train_M,X.txt', 'r')
train_file_list = os.listdir("C:/EIT_284/train/log_1.e4/")
N_dir_train = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/train/0/'
M_dir_train = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/train/M/'
X_dir_train = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/train/X/'

# validation
flare_validation = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_val_M,X.txt', 'r')
val_file_list = os.listdir("C:/EIT_284/validation/log_1.e4/")
N_dir_val = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/val/0/'
M_dir_val = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/val/M/'
X_dir_val = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/val/X/'

# test
flare_test = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_test_M,X.txt', 'r')
test_file_list = os.listdir("C:/EIT_284/test/log_1.e4/")
N_dir_test = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/test/0/'
M_dir_test = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/test/M/'
X_dir_test = 'C:/Projects/keras_talk/JKS/log_1.e4_M,X/test/X/'


lines_train = flare_train.readlines()
lines_val = flare_validation.readlines()
lines_test = flare_test.readlines()

# Constant
dt = timedelta(hours=24)


# Train
for img_name in train_file_list:
    cls=[]
    img_time = datetime(year=int(img_name[8:12]), month=int(img_name[12:14]), day=int(img_name[14:16]),
                          hour=int(img_name[17:19]), minute=int(img_name[19:21]))
    print(img_name)
    for line in lines_train:
        f_time = datetime(year=int(line[:4]), month=int(line[4:6]), day=int(line[6:8]),
                          hour=int(line[9:11]), minute=int(line[12:14]))

        if(f_time>=img_time and f_time <=(img_time+dt)):
            cls.append(line[15:16])
        elif(f_time > img_time+dt):
            break
        # print(img_time, f_time)
    if (cls.count('X')!=0):
        shutil.copy("C:/EIT_284/train/log_1.e4/%s" % img_name, X_dir_train + img_name)
    elif (cls.count('M')!=0):
        shutil.copy("C:/EIT_284/train/log_1.e4/%s" % img_name, M_dir_train + img_name)
    else:
        shutil.copy("C:/EIT_284/train/log_1.e4/%s"%img_name, N_dir_train+img_name)


# Validation
for img_name in val_file_list:
    cls = []
    img_time = datetime(year=int(img_name[8:12]), month=int(img_name[12:14]), day=int(img_name[14:16]),
                        hour=int(img_name[17:19]), minute=int(img_name[19:21]))
    print(img_name)
    for line in lines_val:
        f_time = datetime(year=int(line[:4]), month=int(line[4:6]), day=int(line[6:8]),
                          hour=int(line[9:11]), minute=int(line[12:14]))

        if (f_time >= img_time and f_time <= (img_time + dt)):
            cls.append(line[15:16])
        elif (f_time > img_time + dt):
            break
            # print(img_time, f_time)
    if (cls.count('X') != 0):
        shutil.copy("C:/EIT_284/validation/log_1.e4/%s" % img_name, X_dir_val + img_name)
    elif (cls.count('M') != 0):
        shutil.copy("C:/EIT_284/validation/log_1.e4/%s" % img_name, M_dir_val + img_name)
    else:
        shutil.copy("C:/EIT_284/validation/log_1.e4/%s" % img_name, N_dir_val + img_name)


# Test
for img_name in test_file_list:
    cls=[]
    img_time = datetime(year=int(img_name[8:12]), month=int(img_name[12:14]), day=int(img_name[14:16]),
                          hour=int(img_name[17:19]), minute=int(img_name[19:21]))
    print(img_name)
    for line in lines_test:
        f_time = datetime(year=int(line[:4]), month=int(line[4:6]), day=int(line[6:8]),
                          hour=int(line[9:11]), minute=int(line[12:14]))

        if(f_time>=img_time and f_time <=(img_time+dt)):
            cls.append(line[15:16])
        elif(f_time > img_time+dt):
            break
        # print(img_time, f_time)
    if (cls.count('X')!=0):
        shutil.copy("C:/EIT_284/test/log_1.e4/%s" % img_name, X_dir_test + img_name)
    elif (cls.count('M')!=0):
        shutil.copy("C:/EIT_284/test/log_1.e4/%s" % img_name, M_dir_test + img_name)
    else:
        shutil.copy("C:/EIT_284/test/log_1.e4/%s"%img_name, N_dir_test+img_name)
