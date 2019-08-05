import utils

content_list = utils.read_csv('set100w.csv')
for index, line in enumerate(content_list):
    try:
        if float(line[1]) < 1:
            print('index: {} fare: {}'.format(index, line[1]))
    except:
        print(line)