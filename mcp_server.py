from mcp.server import MCPServer
from lookup_tool import LEAVE_DB

#https://github.com/modelcontextprotocol/python-sdk
mcp = MCPServer("primegate")
# here we make a mc server object called primegate 

@mcp.tool()
def lookup_leave_balance(employee_id: str) -> str:
    """Look up an employee's exact leave balance and status by employee ID."""

    employee_id = employee_id.upper()
    record = LEAVE_DB.get(employee_id)

    if record is None:
        return f"No employee record found for ID '{employee_id}'."

    return (
        f"Employee {record['name']} ({employee_id}): "
        f"{record['leave_balance_days']} leave days remaining, "
        f"status: {record['status']}."
    )


if __name__ == "__main__":
    mcp.run()

# caht.py will run this, this code just sits in the background waiting for a client to connect
# and then it stays on standby