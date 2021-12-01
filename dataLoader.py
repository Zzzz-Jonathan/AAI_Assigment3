def read_data(filename, data_length=13):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            tmp = line.rstrip('\n').split()
            #print()
            line_data = []
            for x in range(data_length):
                if tmp[x] == "?":
                    tmp[x] = 3
                line_data.append(float(tmp[x]))
            data.append(line_data)
    return data
def read_label(filename, label_length=1):
    label = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            tmp = line.rstrip('\n')
            # print()
            label.append(int(tmp))
    return label

#print(len(read_data("traindata.txt")))
#print(read_label("trainlabel.txt"))
