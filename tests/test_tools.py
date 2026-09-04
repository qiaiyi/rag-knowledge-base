from tools import calculator, get_tool_schemas, get_tool_map, ASYNC_TOOL_NAMES


class TestCalculator:
    def test_basic_arithmetic(self):
        assert calculator("1+2*3") == "7"
        assert calculator("(1+2)*3") == "9"
        assert calculator("10/4") == "2.5"
        assert calculator("10%3") == "1"
        assert calculator("-5+3") == "-2"

    def test_rejects_injection(self):
        assert "错误" in calculator("__import__('os').system('ls')")
        assert "错误" in calculator("1;2")
        assert "错误" in calculator("1/0")

    def test_rejects_illegal_chars(self):
        assert "非法字符" in calculator("import os")
        assert "非法字符" in calculator("'a'+'b'")

    def test_power_not_allowed(self):
        # ** 通过了字符白名单，但 AST 求值器应拒绝 Pow 节点
        assert calculator("2**10").startswith("计算错误")

    def test_whitespace_ok(self):
        # 表达式内部的空白不影响求值（注意：前导空格会导致 ast.parse 报缩进错误）
        assert calculator("1 + 2") == "3"


class TestToolRegistry:
    def test_schema_names_match_registry(self):
        schema_names = {s["function"]["name"] for s in get_tool_schemas()}
        assert schema_names == set(get_tool_map().keys())

    def test_schema_structure(self):
        for schema in get_tool_schemas():
            fn = schema["function"]
            assert schema["type"] == "function"
            assert fn["description"]
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"
            assert fn["parameters"]["required"]

    def test_async_markers_valid(self):
        assert ASYNC_TOOL_NAMES.issubset(get_tool_map().keys())
