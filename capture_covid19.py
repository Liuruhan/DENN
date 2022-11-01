# -*- coding: utf-8 -*-
# 从网页采集新冠肺炎统计数据并存入excel文件  仅做演示 Ver：20200220.01
import pandas as pd  # 导入pandas模块库，是读写EXCEL专用，安装时需要同时安装xlrd、xlwt、openpyxl这三个模块库
import requests
import json

# 从QQ网站，找到相关数据
url = "https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5"
# 开始抓取网页数据
data = json.loads(requests.get(url=url).json()["data"])

#我们可以看到JSON数据格式中：lastUpdateTime是数据的最新更新时间；
# areaTree中是全国详细的数据。 所以我们只提取中国部分的数据
china = data["areaTree"][0]["children"]

#现在国内的数据就全部在china变量中了，为了方便绘制地图，将各省份、各地市的数据提取出来备用。
data = []

# 开始获取省级的数据，本处为省级
for i in range(len(china)):
    data.append([
        # 省份，如湖北、广东等省份
        china[i]["name"], \
        # 地市，因为本行为省级别的，所以不需要填写地市名称，以合计二字表示
        "合计", \
        # 省今日新增确诊
        china[i]["today"]["confirm"], \
        # 省累计确诊总数
        china[i]["total"]["confirm"], \
        # 省累计死亡
        china[i]["total"]["dead"], \
        # 省死亡比例
        china[i]["total"]["deadRate"] \
        ])
    #继续向下钻取，获取该省份下，每个地市的数据
    for j in range( len(china[i]["children"])  ):
        # 本处为各地市的数据
        data.append([ \
            # 省份，如湖北、广东等省份
            china[i]["name"], \
            # 地市，如武汉、孝感等地市
            china[i]["children"][j]["name"], \
            # 地市今日新增确诊
            china[i]["children"][j]["today"]["confirm"], \
            # 地市累计确诊总数
            china[i]["children"][j]["total"]["confirm"], \
            # 地市累计死亡
            china[i]["children"][j]["total"]["dead"], \
            # 地市死亡比例
            china[i]["children"][j]["total"]["deadRate"] \
            ])

# list转dataframe 。此处实际就是EXCEL中各列的标题，个数要与上面列的个数相对应
df = pd.DataFrame(data, columns=['地区','城市','今日新增确诊','累计确诊总数','累计死亡','死亡比例'])
print(df)

# 保存到本地excel
df.to_excel("./新冠肺炎明细.xlsx", index=False)
print('新冠肺炎列表已经采集完成并保存到文件【新冠肺炎.xlsx】，程序结束')