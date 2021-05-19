from bs4 import BeautifulSoup
import re
import glob
import os


newFile = open("bs4output/articletext.txt", "w", encoding='utf8')

# Character codedc decode error -> needs fixing, occures when parsing 1articles.html ->

def returnnotags(text):
    p = re.compile(r'<.*?>')
    # print(p.sub('', text))
    return p.sub('', text)


os.chdir('output')

for file in glob.glob('*articles.html'):
    filename = file
    print(filename)
    with open(filename, encoding='utf8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    tags = soup('p')

    for tag in tags:
        print(tag)

    for tag in tags:
        tagex = returnnotags(str(tag))
        if len(tagex) > 2:
            newFile.write(tagex)

# with open("output/1articles.html") as fp:
#     soup = BeautifulSoup(fp, 'html.parser')
#
# tags = soup('p')
#
# for tag in tags:
#     print(tag)
#
# for tag in tags:
#     tagex = returnnotags(str(tag))
#     if len(tagex) > 2:
#         newFile.write(tagex)

newFile.close()