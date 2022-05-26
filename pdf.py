# -*- coding: utf-8 -*-

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextContainer
from pdfminer.pdfpage import PDFPage

import os, sys, re
import glob
import numpy as np
import pandas as pd

def makeDF(l_all):
    l_xs = []; l_ys = []
    for a in l_all:
        if a[-1] > 0:
            flag_x = True
            for x in l_xs:
                if (a[1] < x[1]) and (a[2] > x[0]):
                    x[0] = min(x[0], a[1])
                    x[1] = max(x[1], a[2])
                    flag_x = False
                    break
            if flag_x:
                l_xs.append([a[1], a[2]])

        flag_y = True
        for y in l_ys:
            if (a[3] < y[1]) and (a[4] > y[0]):
                y[0] = min(y[0], a[3])
                y[1] = max(y[1], a[4])
                flag_y = False
                break
        if flag_y:
            l_ys.append([a[3], a[4]])
    l_xs = sorted(l_xs, key=lambda x:x[0])
    l_ys = sorted(l_ys, key=lambda x:-x[0])
    
    tmp = np.zeros(shape=(len(l_ys), len(l_xs)), dtype='int')
    rname = [[] for _ in range(len(l_ys))]

    for a in l_all:
        y = (a[3] + a[4]) / 2
        for yi, (y0, y1) in enumerate(l_ys):
            if y0 < y < y1:
                break
        if a[-1]==0:
            rname[yi].append([a[0], a[1]])
        else:
            x = (a[1] + a[2]) / 2
            for xi, (x0, x1) in enumerate(l_xs):
                if x0 < x < x1:
                    break
            tmp[yi, xi] = int(a[0].replace(",", ""))

    rname = list(map(lambda x:"".join([xx[0] for xx in sorted(x, key=lambda xx:xx[1])]), rname))
    df_tmp = pd.DataFrame(tmp, index=rname)
    df_tmp = df_tmp[~df_tmp.index.str.contains(r"価格帯別戸数")]
    df_tmp.columns = [f"{a}_{b}" for a, b in zip(["発売", "契約"]*6, np.repeat(["首都圏", "都区部", "都下", "神奈川", "埼玉", "千葉"],2))]

    assert((df_tmp.iloc[:23].sum(axis=0) - df_tmp.iloc[23] == 0).all())
    assert((df_tmp.iloc[24:-1].sum(axis=0) - df_tmp.iloc[-1] == 0).all())

    df_tmp = df_tmp[[i != 23 for i in range(df_tmp.shape[0])]]
    df_tmp = df_tmp.stack().to_frame().T
    
    return df_tmp

rsrcmgr = PDFResourceManager()
laparams = LAParams()
laparams.detect_vertical = False
laparams.char_margin = 0.1
laparams.word_margin = 0.1
laparams.line_margin = 0.05

import warnings
warnings.filterwarnings('ignore')

df_all = None
sfiles = sorted([int(s.replace(".pdf", "")) for s in glob.glob('*.pdf')])

# sfiles = filter(lambda x:x > 202107, sfiles)

for sfile in sfiles:
    sys.stdout.write(f"\r{sfile}"); sys.stdout.flush()
    
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with open(f"{sfile}.pdf", 'rb') as fp:
        for page in PDFPage.get_pages(fp, pagenos=[3], maxpages=0, caching=False, check_extractable=False):
            interpreter.process_page(page)
            layout = device.get_result()
            
#             for lt in layout:
#                 print(lt)
#                 break
#             break
                        
            x0_min = 1 << 20
            y1_max = 0
            for lt in layout:
                if isinstance(lt, LTTextContainer):
                    if lt.get_text().find("発") > -1:
                        x0_min = min(lt.x0, x0_min)
                        y1_max = max(lt.y0, y1_max)
            
            l_all = []
            for lt in layout:
                if isinstance(lt, LTTextContainer):
                    if len(lt.get_text().strip())==0: continue
                    if (lt.x0 > x0_min) and (lt.y1 < y1_max):
                        l_all.append([lt.get_text().strip(), lt.x0, lt.x1, lt.y0, lt.y1, 1])
                    elif (lt.y1 < y1_max): 
                        l_all.append([lt.get_text().strip(), lt.x0, lt.x1, lt.y0, lt.y1, 0])
            df_tmp = makeDF(l_all)
            df_tmp.index = [sfile]
            df_all = df_tmp if df_all is None else pd.concat([df_all, df_tmp], axis=0)
    device.close()
#     break
print("\ndone.")

def makeDF2(l_all):
    l_xs = []; l_ys = []
    for a in l_all:
        flag_y = True
        for y in l_ys:
            if (a[3] < y[1]) and (a[4] > y[0]):
                y[0] = min(y[0], a[3])
                y[1] = max(y[1], a[4])
                flag_y = False
                break
        if flag_y:
            l_ys.append([a[3], a[4]])
    l_ys = sorted(l_ys, key=lambda x:-x[0])
    
    rname = [[] for _ in range(len(l_ys))]
    val = [[] for _ in range(len(l_ys))]

    for a in l_all:
        y = (a[3] + a[4]) / 2
        for yi, (y0, y1) in enumerate(l_ys):
            if y0 < y < y1:
                break
        if a[-1]==0:
            rname[yi].append([a[0], a[1]])
        else:
            val[yi].append([a[0], a[1]])

    rname = list(map(lambda x:"".join([xx[0] for xx in sorted(x, key=lambda xx:xx[1])]), rname))
    rname = [re.sub(r"(?:１．|\s)", "", x) for x in rname]
    val = list(map(lambda x:"".join([xx[0] for xx in sorted(x, key=lambda xx:xx[1])]), val))
    df_tmp = pd.DataFrame(val, index=rname)
    
    return df_tmp

import warnings
warnings.filterwarnings('ignore')

df_all = None
sfiles = sorted([int(s.replace(".pdf", "")) for s in glob.glob('*.pdf')])

for sfile in sfiles:
    sys.stdout.write(f"\r{sfile}"); sys.stdout.flush()
    
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    laparams.detect_vertical = False
    laparams.char_margin = 0.1
    laparams.word_margin = 0.1
    laparams.line_margin = 0.05

    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    with open(f"{sfile}.pdf", 'rb') as fp:
        for page in PDFPage.get_pages(fp, pagenos=[2], maxpages=0, caching=True, check_extractable=False):
            interpreter.process_page(page)
            layout = device.get_result()
            
#             for lt in layout:
#                 print(lt)
#             break
                        
            x0_min = 1 << 20
            y0_max = 0
            for lt in layout:
                if isinstance(lt, LTTextContainer):
                    if lt.get_text().find("……") > -1:
                        x0_min = min(lt.x0, x0_min)
                        y0_max = max(lt.y1, y0_max)
            
            l_all = []
            for lt in layout:
                if isinstance(lt, LTTextContainer):
                    if len(lt.get_text().strip())==0: continue
                    if re.search(r"(?:・|……)", lt.get_text()): continue
                    if (lt.x0 > x0_min) and (lt.y0 < y0_max):
                        l_all.append([lt.get_text().strip(), lt.x0, lt.x1, lt.y0, lt.y1, 1])
                    elif (lt.y0 < y0_max): 
                        l_all.append([lt.get_text().strip(), lt.x0, lt.x1, lt.y0, lt.y1, 0])
            
            df_tmp = makeDF2(l_all)
            df_tmp.columns = [sfile]
            df_all = df_tmp if df_all is None else df_all.merge(df_tmp, how='outer', left_index=True, right_index=True)
    device.close()
df_all = df_all.T
print("\ndone.")

df_all['対象物件'] = df_all['対象物件'].str.extract(r"^([0-9]+)物件$").astype(int)
for c in ['発売戸数', '全売却戸数']:
    df_all[c] = df_all[c].str.replace(r",", "", regex=True).str.extract(r"^([0-9]+)戸$").astype(int)
for c in ['１物件当り平均戸数', '１日当り分譲戸数']:
    df_all[c] = df_all[c].str.extract(r"^([0-9.]+)戸$").astype(float)
df_all['総発売額'] = df_all['総発売額'].str.replace(r",", "", regex=True).str.extract(r"^([0-9.]+)万円$").astype(float)
df_all[['分譲単価1平米', '分譲単価3.3平米']] = df_all['１㎡当り分譲単価'].str.extract(r"([0-9.]+)万円\(3\.3㎡当り([0-9.]+)万円\)$")
for c in ['総敷地面積', '総建築面積', '総建築延面積', '総有効分譲面積', '１戸当り平均専有面積']:
    df_all[c] = df_all[c].str.replace(r",", "", regex=True).str.extract(r"^([0-9.]+)㎡$").astype(float)
df_all['総棟数'] = df_all['総棟数'].str.extract(r"^([0-9.]+)棟$").astype(float)
df_all['１棟当り平均階高'] = df_all['１棟当り平均階高'].str.extract(r"^([0-9.]+)階$").astype(float)
df_all['平均所要時間'] = df_all['平均所要時間'].str.extract(r"^最寄駅から([0-9.]+)分$").astype(float)
df_all['１戸当り平均価格'] = df_all['１戸当り平均価格'].str.replace(r",", "", regex=True).str.extract(r"([0-9]+)万円").astype(int)
