"""Ground quantifiers in operator words before sending row missions to the robot."""
import re

from .online_language import CONDITION, NUMBER, measurements, quantity


def integer(word, maximum):
    n = quantity(word)
    if n != int(n) or not 1 <= n <= maximum:
        raise ValueError(f'行号、数量或次数须为 1～{maximum} 的整数')
    return int(n)


def row_filter(q):
    seek = {'target': 'row'}
    sides = [side for char, side in [('左','left'),('右','right')] if char in q]
    if len(sides)>1:
        raise ValueError('需要分开指定左右区域')
    if sides:
        seek.update(side=sides[0],area=True)
    if '对面' in q:
        seek['search_axis']='front'
        if sides:
            seek['region']=['front',sides[0]]
    crop=re.search('草莓|盆栽|生菜|番茄|黄瓜|香草',q)
    if crop:
        seek['label']=crop[0]
    return seek


def collection_plan(text, parse_other):
    q=text.replace('行作物','作物行')
    if not re.search(r'作物行|种植行|行间|排|条.{0,6}行|(?:草莓|盆栽|生菜|黄瓜|香草|番茄)行',q):
        return None
    every=r'所有|全部|每一?条|逐行|逐排|各个|各排|各条|每一?排|每一?行|巡遍'
    number=r'('+NUMBER+r')'
    count_pattern=r'(?:连续|任选|任挑|挑|巡检|巡视|巡查|巡|前|头)\s*'+number+r'\s*(?:条|排|行)'
    if not (re.search(every,q) or re.search(count_pattern,q) or len(re.findall('第'+NUMBER,q))>1):
        return None
    if CONDITION.search(q):
        # Never discard a condition and proceed with a partial collection mission.
        if re.search(every+'|'+count_pattern,q):
            raise ValueError('多行巡检中的条件需要单独确认')
        return None
    parts=[s.strip() for s in re.split('[，,。；;]',q) if s.strip()]
    stages=[]
    first=None
    mission=None
    for part in parts:
        ordinals=[integer(x,30) for x in re.findall('第'+number,part)]
        count=re.search(count_pattern,part)
        all_rows=re.search(every,part)
        repeat=re.search(number+r'\s*(?:遍|趟|圈|次)',part)
        if mission and re.fullmatch(r'每(?:条|排|行).{0,3}'+number+r'(遍|趟|圈|次)',part):
            mission['count']=integer(repeat[1],20)
        elif ordinals and re.search('开始|起步|打头|起点',part) and not count and not all_rows:
            if first is not None:
                raise ValueError('出现多个巡检起点')
            first=dict(row_filter(part),ordinal=ordinals[0])
        elif (count or all_rows or len(ordinals)>1) and re.search('巡|查|巡视|走访|看看',part):
            if mission:
                raise ValueError('请分别指定各批次的巡检范围')
            if re.search('停车|停下|拍照|出去|等待|转弯|左拐|右拐',part):
                raise ValueError('请用逗号分开巡检与后续动作')
            mission=dict(do='patrol_rows',filter=row_filter(part),count=integer(repeat[1],20) if repeat else 1)
            if count:
                start=ordinals[0] if ordinals else 1
                mission['indices']=list(range(start,start+integer(count[1],30)))
            elif len(ordinals)>1:
                mission['indices']=ordinals
            else:
                mission['all']=True
            stages.append(mission)
            if 'speed' in measurements(part):
                mission['speed']=measurements(part)['speed']
        else:
            stages.append(part)
    if mission is None:
        return None
    if first:
        mission['start']=first
    plan=[]
    for stage in stages:
        if isinstance(stage,str):
            plan.extend(parse_other(stage))
        else:
            plan.append(stage)
    return plan
