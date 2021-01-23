import pydantic

d = """{"language_tool" : 3}"""

class R(pydantic.BaseModel):
    language_tool: int = None


r = R.parse_raw(d).dict()
print(r)