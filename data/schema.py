from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class CaseData(BaseModel):
    uid: str = Field(..., description="案件唯一标识符")
    title: str = Field(..., description="案件名称")
    cause: List[str] = Field(default_factory=list, description="案由，用于案例检索粗筛")
    plaintiffs: List[str] = Field(default_factory=list, description="原告列表")
    defendants: List[str] = Field(default_factory=list, description="被告列表")
    plaintiff_claim: str = Field(..., description="原告诉称：包含Root Claim和原告视角事实")
    defendant_argument: Optional[str] = Field(None, description="被告辩称：包含反驳观点和被告视角事实")
    fact_finding: Optional[str] = Field(None, description="审理查明：客观事实描述")
    court_opinion: Optional[str] = Field(None, description="法院观点：用于提取 Insight")
    verdict_result: Optional[str] = Field(None, description="裁判结果：用于判定胜负")
    cited_laws: List[str] = Field(default_factory=list, description="裁判引用的法条，用于验证 A_law")
    def __repr__(self): return f"<CaseData uid={self.uid} cause={self.cause} title={self.title}>"