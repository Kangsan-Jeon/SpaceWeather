from astropy.io import fits
import numpy as np
import pprint
from datetime import date, timedelta

import os

lst=[]

for i in range(2000, 2012):
    path_dir = 'E:/EIT/EIT_284/{}/'.format(i)
    file_list = os.listdir(path_dir)
    for file in file_list:
        if file.find('_13') is not -1:
            lst.append(file[8:16])

d = date(year = 2000, month=1, day=1)
delta_d = timedelta(days=1)
d_end = date(year = 2001, month=1, day=1)



while( d < d_end):
    lst_day=[]
    for day in lst:
        img_date = date(year=int(day[0:4]), month=int(day[4:6]), day=int(day[6:8]))
        if (d.month == img_date.month and d.day == img_date.day):
            lst_day.append(img_date)
    if (len(lst_day) == 12):
        print(lst_day)
    d += delta_d