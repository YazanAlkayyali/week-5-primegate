from langchain.tools import tool

LEAVE_DB={
    "E1001": {"name": "Yazan",  "leave_balance_days": 12, "status": "active"},
    "E1002": {"name": "Rashed", "leave_balance_days": 4,  "status": "active"},
    "E1003": {"name": "Dana",   "leave_balance_days": 0,  "status": "on_leave"},
}


@tool
def lookup_leave_balance(employee_id: str) -> str:
    """Look up an employee's exact leave balance and status by employee ID. Use this when the user asks for a specific, 
    structured fact tied to an ID (e.g. "how many leave days does E1002 have left?"). Do NOT use
    this for general questions about company leave policy, that's a retrieval question, not a lookup question.
    Args:
        employee_id: The employee ID, e.g. "E1001"."""
    
    record=LEAVE_DB.get(employee_id.upper())
    if record is None:
        return f"No employee record found for ID '{employee_id}'."
    return (
        f"Employee {record['name']} ({employee_id.upper()}): "
        f"{record['leave_balance_days']} leave days remaining, "
        f"status: {record['status']}."
    )