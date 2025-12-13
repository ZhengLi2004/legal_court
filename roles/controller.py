from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger
from actions.controller_actions import PlanTactics, VerifyAndDecide
from mas.schema import WorkerInstruction, WorkerReport, WorkerReportStatus
from tools.graph_tool import GraphTool
from tools.initializer import AgentPersona # 引用之前定义的 Persona

class ArgumentController(Role):
    name: str = "Controller"
    profile: str = "Lead Lawyer"
    
    def __init__(self, name: str, persona: AgentPersona, graph_tool: GraphTool, insights: str = ""):
        super().__init__(name=name, profile="Lead Lawyer")
        self.persona = persona
        self.graph_tool = graph_tool
        self.insights = insights
        self.set_actions([PlanTactics, VerifyAndDecide])
        self.round_count = 0 

    async def _act(self) -> Message:
        logger.info(f"{self.name} is thinking...")
        memories = self.get_memories(k=1)
        if not memories or "SYSTEM_START" in memories[0].content: return await self._plan_phase()
        last_msg = memories[0]

        if "REPORT" in str(last_msg.content):
            try:
                report = WorkerReport.model_validate_json(last_msg.content)
                if report.message_type == "REPORT": return await self._decide_phase(report)
            
            except:
                logger.warning("Invalid report format")
                return await self._plan_phase() # 失败则重试计划
            
        return await self._plan_phase()

    async def _plan_phase(self) -> Message:
        context = self.graph_tool.current_graph.latest_context
        last_mem = self.get_memories(k=1)[0]
        feedback = ""

        if "SYSTEM_FEEDBACK" in str(last_mem.content):
            feedback = str(last_mem.content)
            logger.info(f"Controller received feedback: {feedback[:50]}...")

        action = PlanTactics(llm=self.llm)
        
        query_intent = await action.run(
            self.name, 
            self.persona, 
            self.insights, 
            context,
            feedback=feedback
        )
        
        instruction = WorkerInstruction(
            query=query_intent,
            graph_context=context
        )
        
        return Message(
            content=instruction.to_json(), 
            role=self.profile,
            cause_by=PlanTactics,
            send_to="FactWorker"
        )

    async def _decide_phase(self, report: WorkerReport) -> Message:
        context = self.graph_tool.current_graph.latest_context
        
        if report.status == WorkerReportStatus.NOT_FOUND:
            logger.info("Worker found nothing. Re-planning...")
            return await self._plan_phase()
            
        if report.status == WorkerReportStatus.FOUND:
            action = VerifyAndDecide(llm=self.llm)
            
            decision = await action.run(
                self.name, 
                report.content, 
                context, 
                self.persona.initial_strategy
            )
            
            if "ADOPT" in decision:
                final_intent = decision.split("ADOPT:", 1)[1].strip()
                exec_result = await self.graph_tool.process_intent(self.name, final_intent)

                if "REJECT:" in exec_result:
                    logger.warning(f"GraphTool rejected intent: {exec_result}")
                    return await self._plan_phase()

                return Message(content=f"Action Completed: {exec_result}", role=self.profile)

            else:
                logger.info(f"Controller rejected advice: {decision}")
                return await self._plan_phase()
        
        return Message(content="ERROR", role=self.profile)