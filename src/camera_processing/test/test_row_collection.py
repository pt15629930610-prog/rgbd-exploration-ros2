import unittest
from types import SimpleNamespace

from camera_processing.online_language import compile_semantics
from camera_processing.online_skills import Feature, MotionSkills, SearchExhausted, Unmet
from camera_processing.row_collection import collection_plan


class SimulatedGates(MotionSkills):
    pose=(0.,0.,0.)
    course=0.
    now=0.
    def __init__(self):
        self.view=SimpleNamespace(frontier_path=lambda *a,**kw:[])
        self.rows=[Feature('row',(float(i),0.),1.57,approach_yaw=0.) for i in (1,2,3)]
        self.inspections=[]

    def seek_junction(self,step,seen=None):
        yield
        candidates=[f for f in self.rows if f not in (seen or [])]
        if not candidates:
            raise SearchExhausted()
        self.feature=candidates[step.get('ordinal',1)-1]

    def enter(self):
        yield

    def travel(self,point):
        yield

    def run_skill(self,step,next_step):
        self.inspections.append((self.feature.point[0],step['count']))
        yield

    def note(self,msg):
        pass


class CollectionTests(unittest.TestCase):
    def execute(self,q):
        step,=collection_plan(q,lambda s:self.fail(s))
        sim=SimulatedGates()
        list(sim.patrol_rows(step))
        return sim.inspections

    def test_order_and_counts(self):
        self.assertEqual(self.execute('巡第三条和第一条作物行'),[(3,1),(1,1)])
        self.assertEqual(self.execute('巡两条作物行，每条走两遍'),[(1,2),(2,2)])

    def test_start_is_not_filter(self):
        self.assertEqual(self.execute('从左边第二条作物行开始，巡遍所有作物行'),[(2,1),(1,1),(3,1)])

    def test_missing_row_reports_failure(self):
        with self.assertRaises(Unmet):
            self.execute('巡四条作物行')

    def test_straight_to_crossroad(self):
        items=[{'原文':'先直走','动作':'巡检','目标':'作物行'},
               {'原文':'遇到十字路口右拐','动作':'转向','目标':'十字路口','方位':'右'},
               {'原文':'遇到对面第二条行作物进去巡检','动作':'巡检','目标':'作物行'}]
        plan=compile_semantics({'动作序列':items},'，'.join(i['原文'] for i in items))
        self.assertEqual(plan,[{'do':'seek','target':'crossroad'},{'do':'turn','side':'right'},
                               {'do':'seek','target':'row','ordinal':2,'search_axis':'front'},
                               {'do':'enter'},{'do':'patrol'}])
