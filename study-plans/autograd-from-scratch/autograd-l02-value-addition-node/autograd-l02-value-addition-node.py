def value_addition_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns an addition node with id, data, grad, op, and the original parents.
    """
    output_record = {}
    output_record["id"] = output_id
    output_record["data"] = float((left["data"] + right["data"]))
    output_record["grad"] = 0.0
    output_record["op"] = "+"
    output_record["parents"] = [left, right]

    return output_record