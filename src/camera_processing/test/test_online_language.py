"""Ground model plans in the instruction before letting any motion start."""

import unittest
from unittest.mock import patch

from camera_processing.online_language import compile_semantics, interpret_online, measurements


def compile_items(text, *items):
    return compile_semantics({'动作序列': list(items)}, text)


class LanguageTests(unittest.TestCase):
    def test_pasted_build_command_never_starts_motion(self):
        q = ('source /opt/ros/humble/setup.bash && colcon build --packages-select '
             'camera_processing --symlink-install && source install/setup.bash以1米/秒的速度行驶，先右拐')
        with patch('camera_processing.online_language.ask') as model:
            with self.assertRaisesRegex(ValueError, '终端命令'):
                interpret_online('unused', q)
            model.assert_not_called()
        result = compile_items('天气晴朗', {'原文':'天气晴朗','动作':'行驶'})
        self.assertEqual(result[0]['do'], 'unsupported')

    def test_retry_logged_merged_turn_and_traverse(self):
        for text, expected in [('先右拐穿过最后一条行作物', ['turn','seek','enter','traverse_row']),
                               ('穿过最后一条行作物再右拐', ['seek','enter','traverse_row','turn'])]:
            def ask(server, prompt, quote, schema, tokens):
                verb = '转向' if '拐' in quote else '穿过'
                return {'动作序列':[{'原文':quote, '动作':verb, '方位':'右', '目标':'作物行'}
                                if quote == text else {'原文':quote, '动作':verb}]}
            with patch('camera_processing.online_language.ask', side_effect=ask) as model:
                plan = interpret_online('unused', text)
            self.assertIn('各需一个动作', model.call_args_list[1].args[1])
            self.assertEqual([s['do'] for s in plan], expected)
            self.assertEqual(next(s['order'] for s in plan if 'order' in s), 'last')

    def test_junction_response_before_unmarked_destination(self):
        main = '1米/秒的速度行驶，先右拐穿过最后一条行作物'
        cond = '一遇到十字路口就左拐'
        row = '到对面中间那条行作物去巡一圈'
        door = '然后在左门前一米停下'
        text = '，'.join((main, cond, row, door))
        def ask(server, prompt, quote, schema, tokens):
            if quote == text:
                return {}
            clauses = {main: [('1米/秒的速度行驶','调速'), ('先右拐','转向'),
                              ('穿过最后一条行作物','穿过')],
                       row+'，'+door: [(row,'巡检'), (door,'靠近')]}
            return {'动作序列':[{'原文':q,'动作':v} for q,v in clauses[quote]]}
        with patch('camera_processing.online_language.ask', side_effect=ask):
            plan = interpret_online('unused', text)
        self.assertEqual(next(s['watch'] for s in plan if 'watch' in s),
                         dict(effect='turn',target='crossroad',turn_side='left'))
        self.assertEqual([s['order'] for s in plan if 'order' in s], ['last','middle'])
        self.assertEqual(plan[-1]['distance'], 1.)

    def test_relative_row_survives_logged_model_split(self):
        q = '到对面最后一条作物行去巡检'
        steps = compile_items(q, {'原文':q[:-3],'动作':'行驶'},
                              {'原文':'去巡检','动作':'巡检'})
        self.assertEqual(steps, [{'do':'seek','target':'row','search_axis':'front','order':'last'},
                                 {'do':'enter'}, {'do':'patrol'}])
        for noun, order in [('最开始的那条','first'),('中间那条','middle'),('另外一条','other')]:
            q = '到'+noun+'草莓作物行巡两趟'
            steps = compile_items(q, {'原文':q,'动作':'巡检'})
            self.assertEqual((steps[0]['order'],steps[0]['label'],steps[-1]['count']), (order,'草莓',2))
        q = '换另一个巡检'
        steps = compile_items(q, {'原文':q,'动作':'巡检'})
        self.assertEqual(steps, [{'do':'seek','target':'row','order':'other'},
                                 {'do':'enter'}, {'do':'patrol'}])

    def test_scoped_observation_and_recovery_preserve_followup(self):
        from camera_processing.stage_language import PARTS
        main = '以0.3米/秒沿左边第一条草莓行巡一趟'
        condition = '途中看到右门就记下来'
        after = '如果行末堵了就原路返回，巡完后去右门前1米停'
        def ask(server, prompt, text, schema, tokens):
            if '响应' in schema['properties']:
                return dict(zip(PARTS, (main, condition, after)), 目标原话='右门', 响应='记录并继续')
            verb = '巡检' if '巡' in text else '靠近'
            return {'动作序列':[{'原文':text,'动作':verb}]}
        with patch('camera_processing.online_language.ask', side_effect=ask):
            plan = interpret_online('unused', '，'.join((main, condition, after)))
        self.assertEqual([s['do'] for s in plan], ['speed','seek','enter','patrol','seek','approach'])
        self.assertEqual(plan[0]['speed'], .3)
        self.assertEqual((plan[1]['label'],plan[1]['ordinal'],plan[1]['side']), ('草莓',1,'left'))
        self.assertEqual(plan[3]['watch'], dict(effect='record',target='door',side='right'))
        self.assertEqual(plan[-1]['distance'], 1.)

    def test_interruption_scope_and_ambiguous_responses(self):
        from camera_processing.stage_language import scoped_plan, split_rule
        self.assertIsNone(scoped_plan('unused','穿过第一条作物行，然后在十字路口左拐',None))
        self.assertEqual(split_rule('一遇到第二个十字路口就右转'),
                         dict(target='crossroad', ordinal=2, effect='turn', turn_side='right'))
        self.assertEqual(split_rule('期间看到右门就停')['effect'], 'stop')
        for phrase in ('途中没有看到门就停','途中看见门就停三秒','途中遇到右门或十字路口就停'):
            with self.assertRaises(ValueError):
                split_rule(phrase)

    def test_rightward_search_in_far_region_keeps_either_row_orientation(self):
        q = '再到对面往右遇到第二条作物行巡检'
        plan = compile_items(q, {'原文':q, '动作':'巡检', '目标':'作物行'})
        self.assertEqual(plan[0], {'do':'seek', 'target':'row','ordinal':2,
                                   'region':['front'],'search_axis':'right'})

    def test_scoped_speed_and_axis_do_not_constrain_row_orientation(self):
        speed = '全程以0.6米/秒的速度行驶'
        row = '到对面第二条作物行去巡检'
        plan = compile_items(speed+'，'+row, {'原文':speed, '动作':'调速'},
                             {'原文':row, '动作':'巡检','目标':'作物行'})
        self.assertEqual(plan[:2], [{'do':'speed','speed':.6},
                                   {'do':'seek','target':'row','ordinal':2,'search_axis':'front'}])
        door = '去对面门前停下'
        self.assertEqual(compile_items(door, {'原文':door,'动作':'靠近'})[0],
                         {'do':'seek','target':'door','side':'front'})

    def test_door_and_standoff_belong_to_blocked_branch(self):
        q = '穿过第二条作物行'
        with patch('camera_processing.online_language.ask', return_value={
                '动作序列':[{'原文':q,'动作':'穿过','目标':'作物行'}]}):
            plan = interpret_online('unused', q+'，遇到行不通的时候就寻找左边的门，在门前1.5米停下')
        self.assertEqual(plan, [{'do':'seek','target':'row','ordinal':2}, {'do':'enter'},
                               {'do':'traverse_row','if_blocked':[
                                   {'do':'seek','target':'door','side':'left'},
                                   {'do':'approach','distance':1.5}]}])
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
