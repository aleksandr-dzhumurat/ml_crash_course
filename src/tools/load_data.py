import pandas as pd
import urllib.request
from argparse import ArgumentParser
import os

# Adding information about user agent
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

parser = ArgumentParser()

parser.add_argument('-i', '--input', dest='input_file', required=True, help='Input CSV file')
args = parser.parse_args()

df = pd.read_csv(args.input_file)
num_rows = df.shape[0]
cnt = 0
for row in df[['content_id', 'content_url']].values:
    cnt += 1
    filename = '/srv/data/pictures/' + row[0]+'.'+ row[1].split('.')[-1]
    if os.path.exists(filename):
        pass
    else:
        image_url = row[1]
        urllib.request.urlretrieve(image_url, filename)
        print('%d form %d: %d percent' % (cnt, num_rows, int(cnt*1.0 /num_rows*100)))