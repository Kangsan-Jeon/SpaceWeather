def change_date(string):
    if len(string)==10:
        string='0'+string

    if string.count('-Jan-') != 0:
        string = string[7:]+'01'+string[0:2]
    elif string.count('-Feb-') != 0:
        string = string[7:]+'02'+string[0:2]
    elif string.count('-Mar-') != 0:
        string = string[7:] + '03' + string[0:2]
    elif string.count('-Apr-') != 0:
        string = string[7:] + '04' + string[0:2]
    elif string.count('-May-') != 0:
        string = string[7:] + '05' + string[0:2]
    elif string.count('-Jun-') != 0:
        string = string[7:] + '06' + string[0:2]
    elif string.count('-Jul-') != 0:
        string = string[7:] + '07' + string[0:2]
    elif string.count('-Aug-') != 0:
        string = string[7:] + '08' + string[0:2]
    elif string.count('-Sep-') != 0:
        string = string[7:] + '09' + string[0:2]
    elif string.count('-Oct-') != 0:
        string = string[7:] + '10' + string[0:2]
    elif string.count('-Nov-') != 0:
        string = string[7:] + '11' + string[0:2]
    elif string.count('-Dec-') != 0:
        string = string[7:] + '12' + string[0:2]
    return string

def w_line(file_in, file_out):
    file_in.readline()
    file_in.readline()
    file_in.readline()
    file_in.readline()
    file_in.readline()
    for line in file_in:
        line = line.split()
        if len(line) != 0:
            # print(line[0])
            line[0] = change_date(line[0])
            if (line[4][0] == 'C' or line[4][0] == 'M' or line[4][0] == 'X'):
                file_out.write('%s\t%s\t%s\n' % (line[0], line[1], line[4][0]))
        else:
            continue
    return file_out


# Train label
file = open('C:/Users/tks02/OneDrive/문서/우주환경/goes_xray_event_list_2000.txt', 'r')
flare = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_train_.txt', 'w')
w_line(file, flare)

for i in [1,2,3,5,6,7,8,9,10,11]:
    file = open('C:/Users/tks02/OneDrive/문서/우주환경/goes_xray_event_list_20{:02d}.txt'.format(i), 'r')
    flare = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_train_.txt', 'a')
    w_line(file, flare)

# Validation label

file = open('C:/Users/tks02/OneDrive/문서/우주환경/goes_xray_event_list_2004.txt', 'r')
flare = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_val_.txt', 'w')
w_line(file, flare)


# Test label
file = open('C:/Users/tks02/OneDrive/문서/우주환경/goes_xray_event_list_2005.txt', 'r')
flare = open('C:/Users/tks02/OneDrive/문서/우주환경/Flare_test_.txt', 'w')
w_line(file, flare)
