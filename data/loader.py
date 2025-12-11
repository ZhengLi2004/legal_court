import json
import os
from typing import List, Generator, Optional
from pydantic import ValidationError
from .schema import CaseData

class CaseDataLoader:
    def __init__(self, data_dir: str): self.data_dir = data_dir

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        return text.strip()
    
    def _is_valid_field(self, field_value) -> bool:
        if not field_value: return False
        if isinstance(field_value, str) and 'unknown' in field_value.lower(): return False
        return True

    def _parse_single_line(self, line: str) -> Optional[CaseData]:
        try:
            raw = json.loads(line)
            
            if "metaInfo" in raw and "content" in raw:
                meta = raw.get("metaInfo", {})
                content = raw.get("content", {})
                uid = meta.get("uid", "")
                title = meta.get("案件名称")
                cause = meta.get("案由", [])
                plaintiffs = [p.get("pname") for p in meta.get("人物信息", []) if "原告" in p.get("ptypes", [])]
                defendants = [p.get("pname") for p in meta.get("人物信息", []) if "被告" in p.get("ptypes", [])]
                p_claim = self._clean_text(content.get("原告诉称", ""))
                d_arg = self._clean_text(content.get("被告辩称", ""))                
                fact_finding = self._clean_text(content.get("审理查明", ""))
                court_opinion = self._clean_text(content.get("法院观点", ""))
                verdict_result = self._clean_text(content.get("裁判结果", ""))
                cited_laws = meta.get("法律条款", [])

                core_fields_to_check = [
                    uid, title, cause, plaintiffs, defendants,
                    p_claim, d_arg, fact_finding, court_opinion, verdict_result,
                    cited_laws
                ]

                if "辩称" not in d_arg or not all(self._is_valid_field(field) for field in core_fields_to_check): return None
                
                return CaseData(
                    uid=uid, title=title, cause=cause, plaintiffs=plaintiffs,
                    defendants=defendants, plaintiff_claim=p_claim, defendant_argument=d_arg,
                    fact_finding=fact_finding, court_opinion=court_opinion,
                    verdict_result=verdict_result, cited_laws=cited_laws
                )
            
            else: return CaseData(**raw)

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Warning: Skipping line due to parsing error: {e}")
            return None
        
        except Exception as e:
            print(f"Warning: Skipping line due to unexpected error: {e}")
            return None

    def load_all(self, limit: int = None) -> List[CaseData]:
        results = []
        count = 0
        
        if not os.path.exists(self.data_dir): raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not (filename.endswith(".json") or filename.endswith(".jsonl")): continue
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    case = self._parse_single_line(line)
                    
                    if case:
                        results.append(case)
                        count += 1
                        if limit and count >= limit: return results
        
        return results

    def stream(self) -> Generator[CaseData, None, None]:
        if not os.path.exists(self.data_dir): raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not (filename.endswith(".json") or filename.endswith(".jsonl")): continue
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    case = self._parse_single_line(line)
                    if case: yield case