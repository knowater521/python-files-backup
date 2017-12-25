from urllib.request import urlopen
import re
# if has Chinese, apply decode()
html = urlopen(
    "https://morvanzhou.github.io/static/scraping/basic-structure.html"
).read().decode('utf-8')
print(html)


title = re.findall(r'<title>(.+?)</title>',html)

print('title is:',title[0])

res = re.findall(r"<p>(.*?)</p>", html, flags=re.DOTALL)    # re.DOTALL if multi line
print("\nPage paragraph is: ", res[0])
# Page paragraph is:
#     这是一个在 <a href="https://morvanzhou.github.io/">莫烦Python</a>
#     <a href="https://morvanzhou.github.io/tutorials/scraping">爬虫教程</a> 中的简单测试.
#     

res = re.findall(r'href="(.*?)"', html)
print("\nAll links: ", res)