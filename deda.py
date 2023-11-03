import json
import pandas

#
file0 = json.load(open('/home/mzjs/mimic-cxr-master/txt/annotation.json', 'r'))
# # new_file = []
labels = ['train', 'test', 'val']
# # datas = [pandas.DataFrame(columns=['report']) for i in range(6)]
# label_file = pandas.read_csv('/home/mzjs/chexpert-labeler/labeled_reports0.csv')

# i = 0
# j = 0
# for label in labels:
#     for content in file[label]:
#         # path = content['image_path']
#         report = content['report'].replace('\n', '')
#         if report[:10] != label_file.loc[i]['Reports'][:10]:
#             print('[Def err]')
#             print(report)
#             print(label_file.loc[i]['Reports'])
#
#
#         # tag = []
#         # # dic = {'file_name': path[0], 'paragraph': report, 'tag': tag}
#         # # new_file.append(dic)
#         # if i >= 50000:
#         #     j += 1
#         #     i = 0
#         # datas[j].loc[i] = report
#         i += 1
#         if i % 1000 == 0:
#             print(j, i)
# # for k, data in enumerate(datas):
# #     data.to_csv('/home/mzjs/mimic-cxr-master/txt/tag_50000_'+str(k)+'.csv', header=None)
# # json.dump(new_file, open('/home/mzjs/mimic-cxr-master/txt/train_tag.json', 'w'))
# file = pandas.read_csv('/home/mzjs/chexpert-labeler/labeled_reports.csv')
# file0 = pandas.read_csv('/home/mzjs/chexpert-labeler/labeled_reports0.csv')
# file1 = pandas.read_csv('/home/mzjs/chexpert-labeler/labeled_reports1.csv')
# file2 = pandas.read_csv('/home/mzjs/chexpert-labeler/labeled_reports2.csv')
# l = [file0, file1, file2]
# i = 0
# for f in l:
#     for j, item in f.iterrows():
#         file.loc[i+j] = item
#         if j % 1000 == 0:
#             print(i, j)
#     i += 100000
# file.to_csv('/home/mzjs/chexpert-labeler/labeled_reports.csv', header=None)

file1 = pandas.read_csv('/home/mzjs/chexpert-labeler/tags.csv', index_col=0)
tags_l = ['Reports', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                       'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
                       'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
i = 0

for label in labels:
    new_file = []
    for content in file0[label]:
        path = content['image_path']
        report = content['report'].replace('\n', '')
        tags = []
        # for tag in tags_l:
        #     if file1.loc[i][tag] == 1:
        #         tags.append(tag)
        dic = {'file_name': path[0], 'paragraph': report, 'tag': tags}
        new_file.append(dic)
        i += 1
        if i % 5000 == 0:
            print(i)
    json.dump(new_file, open('/home/mzjs/mimic-cxr-master/txt/'+label+'_nt.json', 'w'))