"""Ground model plans in the instruction before letting any motion start."""

import unittest
from unittest.mock import patch

from camera_processing.online_language import compile_semantics, interpret_online, measurements


def compile_items(text, *items):
    return compile_semantics({'动作序列': list(items)}, text)


class LanguageTests(unittest.TestCase):
    def test_patrol_limits_and_reference(self):
        self.assertEqual(measurements('每秒一米，五十米'), {'speed': 1., 'distance': 50.})
        for text in ('每秒1.01米', '五十一米'):
            with self.assertRaises(ValueError):
                measurements(text)
        for q, expected in [
            ('这行一共巡三趟', [{'do':'resume_row'}, {'do':'patrol', 'count':3, 'total':True}]),
            ('这条行再巡两圈', [{'do':'resume_row'}, {'do':'patrol', 'count':2}]),
            ('去右边隔壁的作物行巡一趟', [{'do':'neighbor','side':'right'}, {'do':'enter'}, {'do':'patrol'}]),
        ]:
            self.assertEqual(compile_items(q, {'原文':q, '动作':'巡检', '目标':'作物行'}), expected)
        q = '离第一个十字口还有五米就停'
        self.assertEqual(compile_items(q, {'原文':q, '动作':'靠近'})[-1],
                         {'do':'landmark_stop','distance':5.})
        q = '去第一条作物行巡检绕两圈'
        self.assertEqual(compile_items(q, {'原文':q, '动作':'巡检'})[-1], {'do':'patrol','count':2})

    def test_explicit_distance_still_moves_and_strict_circuit_reuses_the_row(self):
        q = '以0.3米/秒向前走一米'
        self.assertEqual(compile_items(q, {'原文': q, '动作': '行驶'}),
                         [{'do': 'advance', 'side': 'front', 'speed': .3, 'distance': 1.}])
        a, b = '遇到第一条作物行', '进去绕一圈'
        steps = compile_items(a+'，'+b, {'原文': a, '动作': '寻找'}, {'原文': b, '动作': '巡检'})
        self.assertEqual([s['do'] for s in steps], ['seek', 'enter', 'loop'])

    def test_logged_patrol_command(self):
        pairs = [('以0.3米/秒的速度行驶', '行驶'), ('先左拐', '转向'),
                 ('遇到第一条作物行就进去巡一趟', '进入'), ('走不通就退回来', '返回入口'),
                 ('再从右门出去', '穿过')]
        steps = compile_items('，'.join(q for q, _ in pairs), *[
            {'原文': q, '动作': v} for q, v in pairs])
        self.assertEqual(steps, [{'do': 'speed', 'speed': .3}, {'do': 'turn', 'side': 'left'},
                                 {'do': 'seek', 'target': 'row', 'ordinal': 1}, {'do': 'enter'},
                                 {'do': 'patrol'}, {'do': 'seek', 'target': 'door', 'side': 'right'},
                                 {'do': 'exit'}])

    def test_omitted_trigger_or_ordinal_is_recovered_from_the_users_words(self):
        q = '遇到第二个十字路口左拐'
        self.assertEqual(compile_items(q, {'原文': q, '动作': '转向', '方位': '右'}), [
            {'do': 'seek', 'target': 'crossroad', 'ordinal': 2}, {'do': 'turn', 'side': 'left'}])
        q = '先左拐'
        self.assertEqual(compile_items(q, {'原文': q, '动作': '转向', '目标': '十字路口'}),
                         [{'do': 'turn', 'side': 'left'}])

    def test_crossing_before_the_turn_changes_the_trigger(self):
        q = '过了第一个十字路口再左转'
        steps = compile_items(q, {'原文': q, '动作': '转向'})
        self.assertEqual([s['do'] for s in steps], ['seek', 'pass', 'seek', 'enter'])
        self.assertEqual(steps[2], {'do': 'seek', 'target': 'opening', 'side': 'left'})

    def test_spoken_circuit_is_required_even_if_model_just_says_enter(self):
        q = '遇到第一条作物行进去走一圈'
        steps = compile_items(q, {'原文': q, '动作': '进入'})
        self.assertEqual(steps, [{'do': 'seek', 'target': 'row', 'ordinal': 1},
                                 {'do': 'enter'}, {'do': 'loop'}])

    def test_approach_then_dwell_then_return_even_if_two_verbs_are_fused(self):
        q = '去左边的门停留两秒'
        clauses = [{'动作序列': [{'原文': q, '动作': '等待'}]},
                   {'动作序列': [{'原文': '返航到原来的入口', '动作': '返回入口'}]}]
        with patch('camera_processing.online_language.ask', side_effect=clauses):
            steps = interpret_online('unused', q+'后返航到原来的入口')
        self.assertEqual(steps, [{'do': 'seek', 'target': 'door', 'side': 'left'},
                                 {'do': 'approach'}, {'do': 'wait', 'seconds': 2.}, {'do': 'home'}])

    def test_model_cannot_add_duration_or_omit_a_condition(self):
        with self.assertRaises(ValueError):
            compile_items('先左拐，然后等两秒', {'原文': '先左拐', '动作': '转向'})
        with self.assertRaises(ValueError):
            compile_items('先左拐', {'原文': '先左拐两秒', '动作': '转向'})
        q = '如果有虫就拍照'
        self.assertTrue(compile_items(q, {'原文': q, '动作': '拍照'})[0]['dependent'])
        q = '唱首歌'
        self.assertFalse(compile_items(q, {'原文': q, '动作': '等待'})[0]['dependent'])

    def test_speed_distance_and_standoff_units(self):
        self.assertEqual(measurements('以每秒十五厘米往前走半米'), {'speed': .15, 'distance': .5})
        self.assertEqual(measurements('等半分钟'), {'seconds': 30.})
        with self.assertRaises(ValueError):
            measurements('以每秒三米前进')
        q = '到右边的生菜前八十厘米停下'
        steps = compile_items(q, {'原文': q, '动作': '靠近'})
        self.assertEqual(steps[-1], {'do': 'approach', 'label': '生菜', 'distance': .8})

    def test_open_vocabulary_object_and_untargeted_photo(self):
        q = '瞅瞅前头有没有垃圾桶'
        answers = [{'动作序列': [{'原文': q, '动作': '观察', '目标': '通道开口'}]}, {'名称': '垃圾桶'}]
        with patch('camera_processing.online_language.ask', side_effect=answers):
            steps = interpret_online('unused', q)
        self.assertEqual(steps[0], {'do': 'seek', 'target': 'object', 'side': 'front', 'label': '垃圾桶'})
        q = '照张相'
        with patch('camera_processing.online_language.ask', return_value={
                '动作序列': [{'原文': q, '动作': '拍照', '物体名称': q, '目标': '其他物体'}]}):
            self.assertEqual(interpret_online('unused', q), [{'do': 'photo'}])

    def test_region_aisle_and_colour_survive_model_omissions(self):
        q = '从对面左边的道路穿过去'
        steps = compile_items(q, {'原文': q, '动作': '穿过'})
        self.assertEqual(steps[0], {'do': 'seek', 'target': 'opening', 'side': 'left',
                                    'region': ['front', 'left']})
        self.assertEqual(steps[1:], [{'do': 'enter'}, {'do': 'seek', 'target': 'opening'}])
        q = '去右边的棕色门前停车'
        self.assertEqual(compile_items(q, {'原文': q, '动作': '靠近'})[0]['label'], '棕色')
        q = '看看右边第二排草莓'
        self.assertEqual(compile_items(q, {'原文': q, '动作': '观察'})[0],
                         {'do': 'seek', 'target': 'row', 'ordinal': 2, 'side': 'right', 'area': True,
                          'label': '草莓'})
