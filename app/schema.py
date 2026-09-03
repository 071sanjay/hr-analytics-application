from pydantic import BaseModel

class HrSchema(BaseModel):
    Salary:float
    EmpSatisfaction:float
    EngagementSurvey:float
    Absences:float
    DaysLateLast30:float 
    Tenure_years:float
    Age:float
    SpecialProjectsCount:float