"""Defines the Pydantic data model for a legal case.

This module contains the `CaseData` class, which serves as the canonical,
validated data structure for representing a single legal case throughout the
application. By using Pydantic, it ensures that all case data is well-formed
and consistent.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class CaseData(BaseModel):
    """A Pydantic model representing the structured data of a single legal case.

    This class uses Pydantic for data validation and type enforcement, ensuring
    that all instances of a case have a consistent and predictable structure.
    It includes all key sections of a typical legal document, from metadata
    like title and parties to the substantive content like claims, arguments,
    and verdict.

    Attributes:
        uid: The unique identifier for the case.
        title: The name of the case.
        cause: A list of causes of action, used for coarse-grained case retrieval.
        plaintiffs: A list of plaintiff names.
        defendants: A list of defendant names.
        plaintiff_claim: The claims made by the plaintiff.
        defendant_argument: The arguments made by the defendant in response.
        fact_finding: The objective facts of the case as determined by the court.
        court_opinion: The court's reasoning and perspective, useful for insight extraction.
        verdict_result: The final outcome or judgment of the case.
        cited_laws: A list of legal statutes cited by the judge in the verdict.
    """

    uid: str = Field(...)
    title: str = Field(...)
    cause: List[str] = Field(default_factory=list)
    plaintiffs: List[str] = Field(default_factory=list)
    defendants: List[str] = Field(default_factory=list)
    plaintiff_claim: str = Field(...)
    defendant_argument: Optional[str] = Field(None)
    fact_finding: Optional[str] = Field(None)
    court_opinion: Optional[str] = Field(None)
    verdict_result: Optional[str] = Field(None)
    cited_laws: List[str] = Field(default_factory=list)

    def __repr__(self) -> str:
        """Provide a concise, human-readable representation of the CaseData object.

        Returns:
            A string representation including the UID, cause, and title,
            useful for logging and debugging.
        """
        return f"<CaseData uid={self.uid} cause={self.cause} title={self.title}>"
