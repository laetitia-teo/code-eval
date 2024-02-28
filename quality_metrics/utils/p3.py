import ast


def make_solution(puzzle):
    # chooses one solution among the available ones
    header = puzzle['sol_header'].replace('def sol(', 'def g(')
    # body = np.random.choice(puzzle['sol_bodies'])  # choose at random
    body = puzzle['sol_bodies'][0]  # choose the first one, ideal to get
    return '\n'.join([header, body])


def make_puzzle(puzzle, include_docstring=False):
    if include_docstring:
        splitlines = puzzle['sat'].split('\n')
        splitlines.insert(1, puzzle['sol_docstring'])
        puz_str = '\n'.join(splitlines)
    else:
        puz_str = puzzle['sat']
    return puz_str.replace('def sat(', 'def f(')


def parse_puzzle_from_str(s, debug=False):
    error=False
    try:
        functions = [el for el in ast.parse(s).body if isinstance(el, ast.FunctionDef)]
        if len(functions)==2:
            f = ast.unparse(functions[0])
            g = ast.unparse(functions[1])
        else:
            for i in range(len(functions)):
                if 'f(' in ast.unparse(functions[i]):
                    f = ast.unparse(functions[i])
                if 'g(' in ast.unparse(functions[i]):
                    g = ast.unparse(functions[i])
                    
        if not 'f(' in f:
            print("/!\Error in parsing f")
            error=True
        if not 'g(' in g:
            print("/!\Error in parsing g")
            error=True
        if debug:
            return f, g,error
        else:
            return f, g
    except:
        print("/!\Error in parsing")
        if debug:
            return '', '', True
        else:
            return '', ''


def get_puzzle_sol(puzzle):
    if 'program_str' in puzzle:
        return parse_puzzle_from_str(puzzle['program_str'])
    else:  # train set
        return make_puzzle(puzzle), make_solution(puzzle)