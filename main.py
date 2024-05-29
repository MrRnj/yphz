import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import HeatMap, Bar, Sankey
import pandas as pd
from datetime import datetime
from chinese_calendar import is_holiday, is_workday
import numpy as np
import streamlit.components.v1 as components


def format_pre(data):
    st.session_state.link_lst.append(data[0] + '-' + data[2])

def format_after(data):
    if data[3]:
        fst = data[3]
    else:
        fst = data[2]

    if data[1]:
        sed = data[1]
    else:
        sed = data[0]

    st.session_state.link_lst.append(fst + '-' + sed)


def format_time(time: str):
    time_obj = datetime.strptime(time, '%H:%M')
    time_ret = time_obj.hour + time_obj.minute / 60
    return f'{round(time_ret, 2):.2f}'


def is_special_day(date):
    if is_holiday(date) or date.weekday() == 4:
        return True
    else:
        return False


def get_weekday(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    return ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][date.weekday()]


def exchangeData(result: list):

    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i]['source'] == result[j]['target'] and result[i]['target'] == result[j]['source']:
                if int(result[i]['value']) > int(result[j]['value']):
                    result[i]['value'] = result[i]['value'] + \
                        result[j]['value']
                    result[j]['value'] = 0
                else:
                    result[j]['value'] = result[j]['value'] + \
                        result[i]['value']
                    result[i]['value'] = 0

    result = [i for i in result if i['value'] != 0]
    return result


@st.cache_data
def get_raw_data(upload_file):
    data = None

    if upload_file is not None:
        data = pd.read_csv(upload_file, delimiter='\t', header=0)
        data = data.dropna(how='all')
        data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m-%d')

    return data


@st.cache_data
def get_show_data(s_d, e_d, train_type):
    # 筛选出时间范围内的车次
    start_date = datetime.combine(s_d, datetime.min.time())
    end_date = datetime.combine(e_d, datetime.max.time())
    # condition = (data['日期'] >= start_date) and (data['日期'] <= end_date)
    condition = st.session_state.raw_data['日期'].between(start_date, end_date)
    data = st.session_state.raw_data.loc[condition, :]

    # 筛选出选择类型的车次
    condition = data['车次'].str.startswith(tuple(train_type))
    data = data.loc[condition, :]

    return data


def get_yp_data(s_d, e_d, train_type, is_hollyday=False):

    st.session_state['show_data'] = get_show_data(s_d, e_d, train_type)

    data = st.session_state.show_data.drop_duplicates(subset=['日期', '车次'])
    data.dropna(inplace=True, how='all')

    data.replace({'无': 0, '有': 200}, inplace=True)
    data.fillna(0, inplace=True)

    data['车次数字'] = data['车次'].str.extract('[A-Z](\\d+)').astype(int)
    # 增加车次列
    data['车次数'] = 1

    if is_hollyday is True:
        data = data[data['日期'].apply(is_special_day)]

    return data


def get_ls_data():

    dataforspeed = st.session_state.show_data.drop_duplicates(subset=['车次'])
    dataforspeed.sort_values(by='旅程时间', inplace=True)
    dataforspeed['旅程时间'] = dataforspeed['旅程时间'].apply(format_time)

    return dataforspeed


def get_jl_data():

    dataforlink = st.session_state.show_data.drop_duplicates(subset=[
                                                             '车次', '首尾站'])
    dataforlink.dropna(how='all', inplace=True)
    dataforlink.drop(['出发时间', '到站时间', '旅程时间', '商务座特等座', '高级软卧', '动卧', '软座', '一等座',
                     '二等座', '软卧', '无座', '硬卧', '硬座', '日期'], axis=1, inplace=True)

    return dataforlink


def aggregate_YPdata(group, ttype: str):

    name = datetime.strftime(group.name.date(), '%Y-%m-%d')
    group['出发时间'] = pd.to_datetime(group['出发时间'], errors='coerce')
    group['二等座'] = pd.to_numeric(group['二等座'])

    group = group.dropna(subset=['出发时间'])
    group = group.set_index('出发时间')
    # print(f'{name}:{ttype}列车{len(group.index)}对/日')
    group = group.resample(f'{st.session_state.yp_agg_time}Min').max()

    # 将增加的二等座票数行数据设为 -1
    group['二等座'].fillna('-', inplace=True)
    group['二等座'] = group['二等座'].apply(lambda x: float(x) if x != '-' else x)
    group.index = group.index.time
    group = group.rename(columns={'二等座': name})
    st.session_state.yp_lst.append(group)
    # lst.append(group)


def aggregate_PCdata(group):

    name = datetime.strftime(group.name.date(), '%Y-%m-%d')
    group['出发时间'] = pd.to_datetime(group['出发时间'], errors='coerce')

    group = group.dropna(subset=['出发时间'])
    group = group.set_index('出发时间')
    # print(group.resample(f'{cc}Min')['车次数'])
    group = group.resample(f'{st.session_state.pc_agg_time}Min').sum()

    # 将增加的二等座票数行数据设为 -1
    group['车次数'].fillna('-', inplace=True)
    group['车次数'] = group['车次数'].apply(lambda x: float(x) if x != '-' else x)
    group.index = group.index.time
    group = group.rename(columns={'车次数': name})
    st.session_state.pc_lst.append(group)


@st.experimental_fragment()
def format_yp_fig():
    
    is_hollyday = st.toggle('是否为节假日', value=False, key='is_hollyday')
    # st.write(st.session_state.is_hollyday)
    st.session_state.yp_data = get_yp_data(
        st.session_state.date_span[0],
        st.session_state.date_span[1],
        st.session_state.train_type,
        is_hollyday)
   
    if ('yp_out_result' in st.session_state) and (st.session_state.yp_out_result is not None): 
        st.download_button(label= '下载余票结果', data= st.session_state.yp_out_result, file_name='余票结果.csv', mime='text/csv')        

    if 'yp_lst' not in st.session_state:
        st.session_state.yp_lst = []

    st.session_state.yp_lst.clear()
    # 根据日期进行分组并提取所需列数据
    grouped_data = st.session_state.yp_data.groupby('日期')[['出发时间', '二等座']]
    # 对每个分组应用聚合函数
    grouped_data.apply(aggregate_YPdata, ttype='全部')
    # st.write(st.session_state.yp_lst)
    if len(st.session_state.yp_lst) > 0:
        result = pd.concat(st.session_state.yp_lst, axis=1)
        result.fillna('-', inplace=True)
        result = result.sort_index(ascending=False)
        st.session_state.yp_out_result = result.sort_index(ascending=True).to_csv()

        h = str(len(result.index) * 22 + 200)+'px'
        w = str(len(result.columns) * 30 if len(result.columns) * 30 > 400 else 400)+'px'
        res = result.to_numpy()

        values = [[i, j, res[j, i]]
                  for i in range(res.shape[1]) for j in range(res.shape[0])]
        # init_opts=opts.InitOpts(width=w, height=h)
        heatmap_yp = (HeatMap(init_opts=opts.InitOpts(height=h, width=w))
                      .add_xaxis([f'{d}({get_weekday(d)})' for d in result.columns.to_list()])
                      .add_yaxis('', [d.strftime('%H:%M') for d in result.index.to_list()], values)
                      .set_series_opts(itemstyle_opts=opts.ItemStyleOpts(border_color='#eee', border_width=2),
                                       label_opts=opts.LabelOpts(is_show=False, position='inside', formatter='{c}'))
                      .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=200, is_piecewise=True, is_show=True, orient="horizontal",
                                              pieces=[
                                                  {"min": 0, "max": 0,
                                                   "color": "#FF0000", "label": "无票"},
                                                  {"min": 1, "max": 20,
                                                   "color": "#FF7744", "label": "紧张"},
                                                  {"min": 20, "max": 200,
                                                   "color": "#66FF66", "label": "充足"},
                                              ], pos_left='5px', pos_top='10px'),
            datazoom_opts=[opts.DataZoomOpts(type_="slider", orient="horizontal"),  # 添加水平方向的数据缩放
                           opts.DataZoomOpts(type_="inside"),  # 添加内置的数据缩放
                           ],
            toolbox_opts=opts.ToolboxOpts(pos_left='90%',
                                          item_size=30,
                                          orient='vertical',
                                          feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(background_color='white'),
                                                                          data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                                              is_show=False),
                                              magic_type=opts.ToolBoxFeatureMagicTypeOpts(
                                              is_show=False),
                                              data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False))

                                          ))
                      .render_embed()
                      )

        components.html(heatmap_yp, height=len(
            result.index) * 22 + 300, scrolling=True)
    else:
        st.write('无数据！')


def format_pc_fig():

    if ('pc_out_result' in st.session_state) and (st.session_state.pc_out_result is not None):
            st.download_button(label= '下载频次结果', data= st.session_state.pc_out_result, file_name='频次结果.csv', mime='text/csv')  
    
    st.session_state.yp_data = get_yp_data(
        st.session_state.date_span[0],
        st.session_state.date_span[1],
        st.session_state.train_type,
        False)

    if 'pc_lst' not in st.session_state:
        st.session_state.pc_lst = []

    st.session_state.pc_lst.clear()
    # 根据日期进行分组并提取所需列数据
    grouped_data = st.session_state.yp_data.groupby('日期')[['出发时间', '车次数']]
    # 对每个分组应用聚合函数
    grouped_data.apply(aggregate_PCdata)
    if len(st.session_state.pc_lst) > 0:
        result = pd.concat(st.session_state.pc_lst, axis=1)
        result.fillna('-', inplace=True)
        result = result.sort_index(ascending=False)
        st.session_state.pc_out_result =result.sort_index(ascending=True).to_csv()

        h = str(len(result.index) * 22 + 200) + 'px'
        w = str(len(result.columns) * 30 if len(result.columns) * 30 > 400 else 400) + 'px'
        res = result.to_numpy()

        values = [[i, j, res[j, i]]
                  for i in range(res.shape[1]) for j in range(res.shape[0])]

        heatmap_pc = (HeatMap(init_opts=opts.InitOpts(height=h, width=w))
                      .add_xaxis([f'{d}({get_weekday(d)})' for d in result.columns.to_list()])
                      .add_yaxis('', [d.strftime('%H:%M') for d in result.index.to_list()], values)
                      .set_series_opts(itemstyle_opts=opts.ItemStyleOpts(border_color='#eee', border_width=2),
                                       label_opts=opts.LabelOpts(is_show=True, position='inside'))
                      .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=200, is_piecewise=True, is_show=True, orient="horizontal",
                                              pieces=[
                                                  {"min": 0, "max": 0,
                                                   "color": "#40de5a", "label": "无车次"},
                                                  {"min": 1, "max": 3,
                                                   "color": "#fff143", "label": "1-3对"},
                                                  {"min": 4, "max": 6,
                                                   "color": "#ffa631", "label": "4-6对"},
                                                  {"min": 7, "max": 10,
                                                   "color": "#ff7500", "label": "7-10对"},
                                                  {"min": 11, "max": 20,
                                                   "color": "#f00056", "label": "11-20对"},
                                              ], pos_left='5px', pos_top='10px'),
            datazoom_opts=[opts.DataZoomOpts(type_="slider", orient="horizontal"),  # 添加水平方向的数据缩放
                           opts.DataZoomOpts(
                type_="inside"),  # 添加内置的数据缩放
            ],
            toolbox_opts=opts.ToolboxOpts(pos_left='90%',
                                          item_size=30,
                                          orient='vertical',
                                          feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(background_color='white'),
                                                                          data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                                              is_show=False),
                                              magic_type=opts.ToolBoxFeatureMagicTypeOpts(
                                              is_show=False),
                                              data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False))

                                          ))
            .render_embed()
        )

        components.html(heatmap_pc, height=len(
            result.index) * 22 + 300, scrolling=True)
    else:
        st.write('无数据！')


def format_ls_fig():

    dataforspeed = get_ls_data()
    train_count = len(list(dataforspeed['车次']))
    max_speed = dataforspeed['旅程时间'].max()
    min_speed = dataforspeed['旅程时间'].min()
    avrage_speed = np.mean([float(i)
                           for i in list(dataforspeed['旅程时间'].values)])
    h = '600px'
    w = str(train_count * 30 if train_count * 30 > 550 else 550) + 'px'
    bar = (Bar(init_opts=opts.InitOpts(height=h, width=w))
           .add_xaxis(list(dataforspeed['车次']))
           .add_yaxis('旅行时间', list(dataforspeed['旅程时间']))
           .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name='车次', axislabel_opts=opts.LabelOpts(rotate=90)),
        yaxis_opts=opts.AxisOpts(name='旅行时间(小时)'),
        graphic_opts=[
            opts.GraphicText(
                graphic_item=opts.GraphicItem(left=60, bottom=0, z=100),
                graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                    text=f'列车对数：{train_count}对；最高速度：{float(max_speed):.1f}h；最低速度：{float(min_speed):.1f}h；平均速度：{avrage_speed:.1f}h',
                    font='bold 15px 等线',
                    graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill='blue')))]
    )
        .set_series_opts(
        markline_opts=opts.MarkLineOpts(data=[
            opts.MarkLineItem(
                type_="max",
                name="最大值",
                linestyle_opts=opts.LineStyleOpts(
                    type_="dashed",
                    color='red',
                    width=2)),
            opts.MarkLineItem(
                type_="min",
                name="最小值",
                linestyle_opts=opts.LineStyleOpts(
                    type_="dashed",
                    color='yellow',
                    width=2)),
            opts.MarkLineItem(
                type_="average",
                name="平均值",
                linestyle_opts=opts.LineStyleOpts(
                    type_="dashed", color='green', width=2))]),
        label_opts=opts.LabelOpts(is_show=False)
    )
        .render_embed()
    )

    components.html(bar, height=650, scrolling=True)


def format_jl_fig():

    dataforlink = get_jl_data()
    dataforlink = dataforlink.drop(dataforlink[(dataforlink['起终点站'].str.endswith('-'))
                                               | (dataforlink['首尾站'].str.endswith('-'))]
                                   .index)

    if 'link_lst' not in st.session_state:
        st.session_state.link_lst = []

    st.session_state.link_lst.clear()
    dataforlink['middle'] = dataforlink['起终点站'].str.split(
        '-') + dataforlink['首尾站'].str.split('-')
    dataforlink['middle'].apply(format_pre)
    dataforlink['前序'] = st.session_state.link_lst

    st.session_state.link_lst.clear()
    dataforlink['middle'].apply(format_after)
    dataforlink['后续'] = st.session_state.link_lst

    # 统计前序数据
    qxdf = pd.DataFrame()
    qxdf['前序'] = dataforlink['前序'].copy(deep=True)
    # 删除起终点一致的行
    qxdf = qxdf.drop(qxdf[qxdf['前序'].str.split('-').str[0]
                     == qxdf['前序'].str.split('-').str[1]].index)
    # 按分组重新生成新的df
    qxdf = qxdf.groupby('前序').size().reset_index(name='动车列数')
    # 生成前序列表
    qxResult = []
    for row in qxdf.itertuples():
        qxResult.append({"source": row[1].split(
            '-')[0], "target": row[1].split('-')[1], "value": row[2]})
    # print(qxResult)

    # 统计中序数据
    zxdf = pd.DataFrame()
    zxdf['中序'] = dataforlink['首尾站'].copy(deep=True)
    # 删除终点为-的行
    zxdf = zxdf.drop(zxdf[zxdf['中序'].str.endswith('-')].index)
    # 按分组重新生成新的df
    zxdf = zxdf.groupby('中序').size().reset_index(name='动车列数')
    # 生成中序列表
    zxResult = []
    for row in zxdf.itertuples():
        zxResult.append({"source": row[1].split(
            '-')[0], "target": row[1].split('-')[1], "value": row[2]})
    # print(zxResult)

    # 统计后续数据
    hxdf = pd.DataFrame()
    hxdf['后续'] = dataforlink['后续'].copy(deep=True)
    # 删除起终点一致的行
    hxdf = hxdf.drop(hxdf[hxdf['后续'].str.split('-').str[0]
                     == hxdf['后续'].str.split('-').str[1]].index)
    # 按分组重新生成新的df
    hxdf = hxdf.groupby('后续').size().reset_index(name='动车列数')
    # 生成前序列表
    hxResult = []
    for row in hxdf.itertuples():
        hxResult.append({"source": row[1].split(
            '-')[0], "target": row[1].split('-')[1], "value": row[2]})
    # print(hxResult)

    removePre = []
    removeMid = []
    removeAft = []

    # 前续与中续比较
    for d1 in qxResult:
        for d2 in zxResult:
            if d1['source'] == d2['target']:
                if removePre.count(d1) == 0:
                    removePre.append(d1)

    # 前续与后续比较
    for d1 in qxResult:
        for d2 in hxResult:
            if d1['source'] == d2['target']:
                if (int(d1['value']) > int(d2['value'])):
                    if removeAft.count(d2) == 0:
                        removeAft.append(d2)
                else:
                    if removePre.count(d1) == 0:
                        removePre.append(d1)
    # 中续与后续比较
    for d1 in zxResult:
        for d2 in hxResult:
            if d1['source'] == d2['target']:
                if (int(d1['value']) > int(d2['value'])):
                    if removeAft.count(d2) == 0:
                        removeAft.append(d2)
                else:
                    if removeMid.count(d1) == 0:
                        removeMid.append(d1)

    for i in removePre:
        qxResult.remove(i)

    for i in removeMid:
        zxResult.remove(i)

    for i in removeAft:
        hxResult.remove(i)

    qxResult = exchangeData(qxResult)
    hxResult = exchangeData(hxResult)

    total = qxResult + zxResult + hxResult

    st.session_state.link_lst.clear()
    for i in total:
        st.session_state.link_lst.append(i['source'])
    for i in total:
        st.session_state.link_lst.append(i['target'])

    node_lst = [{'name': i} for i in list(set(st.session_state.link_lst))]
    # # print(node_lst)
    # print("{:<10}{:<10}{:>5}".format('源站', '目的站', '次数'))
    # for d in total:
    #     print("{:<10}{:<10}{:>5}".format(d['source'], d['target'], d['value']))

    sankey_all = (
        Sankey(init_opts=opts.InitOpts(width='1500px', height='1800px'))
        .add(
            "",
            node_lst,
            total,
            linestyle_opt=opts.LineStyleOpts(
                opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
            tooltip_opts=opts.TooltipOpts(trigger_on="mousemove"),
        )
        .set_global_opts(toolbox_opts=opts.ToolboxOpts(pos_left='90%',
                                                       item_size=30,
                                                       orient='vertical',
                                                       feature=opts.ToolBoxFeatureOpts(save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(background_color='white'),
                                                                                       data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                                                           is_show=False),
                                                           magic_type=opts.ToolBoxFeatureMagicTypeOpts(
                                                           is_show=False),
                                                           data_view=opts.ToolBoxFeatureDataViewOpts(is_show=False))

                                                       ))
        .render_embed()
    )

    components.html(sankey_all, height=1800, scrolling=True)


def format_config():
    with st.form('config_form'):
        st.write(f"文件包括{st.session_state.raw_data.shape[0]}条记录")

        start_date = st.session_state.raw_data.loc[0, '日期']
        end_date = st.session_state.raw_data.loc[st.session_state.raw_data.shape[0]-1, '日期']
        # 求start_date与end_date之间的天数
        days = (end_date - start_date).days

        from_date = start_date

        if days >= 7:
            to_date = from_date + pd.Timedelta(days=7)
        else:
            to_date = from_date + pd.Timedelta(days=days)

        st.session_state.date_span = st.date_input(
            "选择数据时间范围", (from_date, to_date), start_date, end_date, format='YYYY.MM.DD')

        st.session_state.train_type = st.multiselect("选择车次类型", ["G", "D", "T", "Z", "C", "S", "K"],
                                                     ['G', 'D', 'C', 'S'],
                                                     format_func=format_train_type)
        st.session_state.yp_agg_time = st.number_input(
            "选择列车余票聚合时间段（分钟）", value=60, step=5)
        st.session_state.pc_agg_time = st.number_input(
            "选择列车频次聚合时间段（分钟）", value=60, step=5)
        st.form_submit_button("提交")

    # st.write(st.session_state.date_span)
    # st.write(st.session_state.train_type)
    # st.write(st.session_state.yp_agg_time)
    # st.write(st.session_state.pc_agg_time)


def format_train_type(ops):
    if ops == "G":
        return "高铁"
    elif ops == "D":
        return "动车"
    elif ops == "T":
        return "特快"
    elif ops == "Z":
        return "直达"
    elif ops == "C":
        return "城际"
    elif ops == "S":
        return "市域"
    elif ops == "K":
        return "快速"


def define_traininfo_tools():

    st.write('<center><h2>车次信息数据分析</h2></center>', unsafe_allow_html=True)
    upload_file = st.file_uploader("请选择车次信息文件", accept_multiple_files=False)
    config_column, fig_column = st.columns([4, 10])

    with config_column:
        if upload_file is not None:
            st.session_state['raw_data'] = get_raw_data(upload_file)
            format_config()
        else:
            st.write("未选择文件")

    if 'raw_data' in st.session_state:
        with fig_column:
            tab1, tab2, tab3, tab4 = st.tabs(
                [" 余票分析 ", " 发车频次分析 ", " 旅速旅速分析 ", " 径路分析 "])
            with tab1:
                if st.session_state['raw_data'] is not None:
                    format_yp_fig()
            with tab2:
                if st.session_state['raw_data'] is not None:
                    format_pc_fig()
            with tab3:
                if st.session_state['raw_data'] is not None:
                    format_ls_fig()
            with tab4:
                if st.session_state['raw_data'] is not None:
                    format_jl_fig()


st.set_page_config(layout="wide", page_title="数据分析及可视化工具集")
if 'current_tool' not in st.session_state:
    st.session_state['current_tool'] = ''

# 获取查询参数
query_params = st.query_params

if 'tool_type' in query_params:
    t_type = query_params['tool_type']
    if t_type == 'traininfo':
        # if st.session_state.current_tool != 'traininfo':
        #     st.session_state.current_tool = 'traininfo'
        define_traininfo_tools()
    else:
        st.title("该工具尚在开发中！！")
else:
    st.title("请选择工具")
