import ast

def instrument_test_code(test_code: str):
    """
    Takes HumanEval test code containing a check(candidate) function,
    rewrites its assert statements so they record pass/fail counts.
    Returns:
        module_globals: dict with executed environment (contains instrumented check)
        results: dict tracking each assertion outcome
    """

    results = {"passed": 0, "failed": 0, "details": []}

    # Parse test code into AST
    tree = ast.parse(test_code)

    class AssertTransformer(ast.NodeTransformer):
        def visit_Assert(self, node: ast.Assert):
            # Create code equivalent to:
            #
            # try:
            #     assert <test>
            #     results["passed"] += 1
            # except AssertionError:
            #     results["failed"] += 1
            #     results["details"].append("assert failed at line X")
            #
            new_node = ast.Try(
                body=[
                    node,  # original assertion
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id="results", ctx=ast.Load()),
                                attr="__setitem__", ctx=ast.Load()
                            ),
                            args=[
                                ast.Constant("passed"),
                                ast.BinOp(
                                    left=ast.Subscript(
                                        value=ast.Name(id="results", ctx=ast.Load()),
                                        slice=ast.Constant("passed"),
                                        ctx=ast.Load(),
                                    ),
                                    op=ast.Add(),
                                    right=ast.Constant(1),
                                )
                            ],
                            keywords=[]
                        )
                    )
                ],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="AssertionError", ctx=ast.Load()),
                        name=None,
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id="results", ctx=ast.Load()),
                                        attr="__setitem__",
                                        ctx=ast.Load(),
                                    ),
                                    args=[
                                        ast.Constant("failed"),
                                        ast.BinOp(
                                            left=ast.Subscript(
                                                value=ast.Name(id="results", ctx=ast.Load()),
                                                slice=ast.Constant("failed"),
                                                ctx=ast.Load(),
                                            ),
                                            op=ast.Add(),
                                            right=ast.Constant(1),
                                        )
                                    ],
                                    keywords=[]
                                )
                            ),
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Subscript(
                                            value=ast.Name(id="results", ctx=ast.Load()),
                                            slice=ast.Constant("details"),
                                            ctx=ast.Load(),
                                        ),
                                        attr="append",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(
                                        f"Assertion failed on line {node.lineno}"
                                    )],
                                    keywords=[]
                                )
                            ),
                        ]
                    )
                ],
                orelse=[],
                finalbody=[]
            )
            return ast.copy_location(new_node, node)

    # Transform AST
    new_tree = AssertTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)

    # Prepare isolated environment
    env = {"results": results}

    # Execute instrumented test code
    exec(compile(new_tree, filename="<ast>", mode="exec"), env)

    return env, results
