import json
import unittest

from power_vln.models import (
    CandidateGoal,
    Pose2D,
    RiskFinding,
    RiskLevel,
    SemanticPlan,
    semantic_plan_from_dict,
)
from power_vln.plan_gate import GateCode, GateContext, evaluate_plan
from power_vln.state_machine import DualSystemStateMachine, SystemEvent, SystemState


def sample_plan():
    goal = CandidateGoal(
        goal_id="cabinet-07-approach",
        pose=Pose2D(x=5.2, y=3.1, yaw=3.14),
        approach_yaw=3.14,
        stop_distance_m=0.8,
        source="semantic_scene_graph",
        confidence=0.91,
    )
    finding = RiskFinding(
        rule_id="route-in-allowed-area",
        level=RiskLevel.HARD_STOP,
        passed=True,
        evidence="candidate path remains inside the approved corridor",
    )
    return SemanticPlan(
        mission_id="inspection-001",
        current_subtask="approach cabinet 07",
        candidates=(goal,),
        selected_goal_id=goal.goal_id,
        findings=(finding,),
        confidence=0.88,
        valid_for_ms=5000,
        replan_triggers=("path_blocked", "target_state_changed"),
        audit_summary="Selected the only reachable and rule-compliant approach pose.",
    )


class SemanticPlanTests(unittest.TestCase):
    def test_json_round_trip(self):
        plan = sample_plan()
        decoded = semantic_plan_from_dict(json.loads(plan.to_json()))
        self.assertEqual(decoded, plan)
        self.assertEqual(decoded.selected_goal.goal_id, "cabinet-07-approach")
        self.assertFalse(decoded.has_hard_stop)

    def test_selected_goal_must_exist(self):
        plan = sample_plan()
        with self.assertRaises(ValueError):
            SemanticPlan(
                mission_id=plan.mission_id,
                current_subtask=plan.current_subtask,
                candidates=plan.candidates,
                selected_goal_id="missing",
                confidence=plan.confidence,
                valid_for_ms=plan.valid_for_ms,
            )


class PlanGateTests(unittest.TestCase):
    def test_valid_plan_is_approved(self):
        decision = evaluate_plan(
            sample_plan(),
            GateContext(
                plan_age_ms=100,
                rules_available=True,
                transform_available=True,
                selected_goal_reachable=True,
            ),
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.codes, (GateCode.APPROVED,))

    def test_gate_collects_all_failures(self):
        decision = evaluate_plan(
            sample_plan(),
            GateContext(
                plan_age_ms=5000,
                rules_available=False,
                transform_available=False,
                selected_goal_reachable=False,
            ),
            minimum_confidence=0.90,
        )
        self.assertFalse(decision.approved)
        self.assertEqual(
            set(decision.codes),
            {
                GateCode.RULES_UNAVAILABLE,
                GateCode.LOW_CONFIDENCE,
                GateCode.EXPIRED,
                GateCode.FRAME_UNAVAILABLE,
                GateCode.UNREACHABLE,
            },
        )


class StateMachineTests(unittest.TestCase):
    def test_normal_plan_execute_cycle(self):
        machine = DualSystemStateMachine()
        machine.dispatch(SystemEvent.MISSION_RECEIVED)
        self.assertEqual(machine.state, SystemState.SLOW_PLANNING)
        machine.dispatch(SystemEvent.PLAN_ACCEPTED)
        self.assertEqual(machine.state, SystemState.FAST_EXECUTING)
        machine.dispatch(SystemEvent.SUBGOAL_REACHED)
        self.assertEqual(machine.state, SystemState.SLOW_PLANNING)

    def test_emergency_preempts_execution(self):
        machine = DualSystemStateMachine()
        machine.dispatch(SystemEvent.MISSION_RECEIVED)
        machine.dispatch(SystemEvent.PLAN_ACCEPTED)
        machine.dispatch(SystemEvent.EMERGENCY)
        self.assertEqual(machine.state, SystemState.SAFETY_OVERRIDE)
        machine.dispatch(SystemEvent.EMERGENCY_CLEARED)
        self.assertEqual(machine.state, SystemState.REPLANNING)

    def test_fault_requires_reset(self):
        machine = DualSystemStateMachine()
        machine.dispatch(SystemEvent.SYSTEM_FAULT)
        self.assertEqual(machine.state, SystemState.FAULT)
        with self.assertRaises(ValueError):
            machine.dispatch(SystemEvent.MISSION_RECEIVED)
        machine.dispatch(SystemEvent.RESET)
        self.assertEqual(machine.state, SystemState.IDLE)


if __name__ == "__main__":
    unittest.main()
