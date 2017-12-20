import re
import math
import pprint

#--- utility ------------------------------------------------------------------#
def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines

def manhattan_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return abs(x1 - x2) + abs(y1 - y2)

def neighbour_weight(matrix, x, y):
    weight = 0
    for x_ in range(-1, 2):
        for y_ in range(-1, 2):
            if not (x_ == 0 and y_ == 0):
                if not (x+x_ < 0 or x+x_ >= len(matrix) or y+y_ < 0 or y+y_ >= len(matrix)):
                    weight += matrix[x+x_][y+y_]
    return weight

def print_matrix(matrix, pos=None):
    dim = len(matrix)
    for y in range(dim):
        for x in range(dim):
            if pos == None:
                print("%6d " % matrix[x][y], end='')
            else:
                if x == pos[0] and y == pos[1]:
                    print("%6s " % '(_)', end='')
                else:
                    print("%6s " % matrix[x][y], end='')
        print()


#------------------------------------------------------------------------------#
def day1():
    def captcha(num):
        if type(num) == int:
            numbers = [int(s) for s in str(num)]
        else:
            numbers = [int(s) for s in num]
        sum_ = 0
        for i in range(0, len(numbers)-1):
            if numbers[i] == numbers[i+1]:
                sum_ += numbers[i]
        if numbers[-1] == numbers[0]:
            sum_ += numbers[-1]
        return sum_

    assert captcha(1122) == 3
    assert captcha(1111) == 4
    assert captcha(1234) == 0
    assert captcha(91212129) == 9

    for line in read_file('day1.txt'):
        print("day1:", captcha(line))


def day1_part_two():
    def captcha(num):
        if type(num) == int:
            numbers = [int(s) for s in str(num)]
        else:
            numbers = [int(s) for s in num]
        sum_ = 0
        len_ = len(numbers)
        for i in range(0, len_):
            if numbers[i] == numbers[int(i+len_/2) % len_]:
                sum_ += numbers[i]
        return sum_

    assert captcha(1212) ==  6
    assert captcha(1221) ==  0
    assert captcha(123425) ==  4
    assert captcha(123123) ==  12
    assert captcha(12131415) ==  4

    for line in read_file('day1.txt'):
        print("day1_part_two:", captcha(line))


#------------------------------------------------------------------------------#
def day2():
    def checksum(matrix):
        checksum_ = 0
        for row in matrix:
            checksum_ += max(row) - min(row)
        return checksum_

    test_ = [ [5, 1, 9, 5],
              [7, 5, 3],
              [2, 4, 6, 8] ]
    assert checksum(test_) == 18

    matrix = list()
    for line in read_file('day2.txt'):
        matrix.append([int(x) for x in re.split(" |\t", line)])
    print("day2:", checksum(matrix))


def day2_part_two():
    def checksum(matrix):
        checksum_ = 0
        for row in matrix:
            for num in row:
                for denom in row:
                    if num != denom and num % denom == 0:
                        checksum_ += int(num / denom)
        return checksum_

    test_ = [ [5, 9, 2, 8],
              [9, 4, 7, 3],
              [3, 8, 6, 5] ]
    assert checksum(test_) == 9

    matrix = list()
    for line in read_file('day2.txt'):
        matrix.append([int(x) for x in re.split(" |\t", line)])
    print("day2_part_two:", checksum(matrix))


#------------------------------------------------------------------------------#
def day3():

    def get_dim(num):
        i = 1
        while num > i*i:
            i += 2
        return i

    def coord(num, dim):
        x, y = dim, dim-1
        offset = num - (dim-2)*(dim-2) - 1

        if offset == 0:
            return (x, y)
        while y > 1 and offset > 0:
            y -= 1
            offset -= 1
        if offset == 0:
            return (x, y)
        while x > 1 and offset > 0:
            x -= 1
            offset -= 1
        if offset == 0:
            return (x, y)
        while y < dim and offset > 0:
            y += 1
            offset -= 1
        if offset == 0:
            return (x, y)
        while x < dim and offset > 0:
            x += 1
            offset -= 1
        return (x, y)

    def dist(num):
        dim = get_dim(num)
        center = (int(dim/2 + 1), int(dim/2 + 1))
        pos = coord(num, dim)
        return manhattan_distance(pos, center)

    assert dist(8) == 1
    assert dist(24) == 3
    assert dist(26) == 5
    assert dist(27) == 4
    assert dist(48) == 5
    assert dist(49) == 6

    print("day3:", dist(325489))


def day3_part_two():
    def traverse(matrix, num):
        dim = len(matrix)
        matrix[int(dim/2)][int(dim/2)] = 1 # center
        # startpos
        x, y = int(dim/2+1), int(dim/2+1)
        for step in range(2, dim, 2):
            # north"
            for s in range(0, step):
                y -= 1
                matrix[x][y] = neighbour_weight(matrix, x, y)
                if matrix[x][y] > num:
                    return matrix[x][y]
            # west"
            for s in range(0, step):
                x -= 1
                matrix[x][y] = neighbour_weight(matrix, x, y)
                if matrix[x][y] > num:
                    return matrix[x][y]
            # south"
            for s in range(0, step):
                y += 1
                matrix[x][y] = neighbour_weight(matrix, x, y)
                if matrix[x][y] > num:
                    return matrix[x][y]
            # east"
            for s in range(0, step):
                x += 1
                matrix[x][y] = neighbour_weight(matrix, x, y)
                if matrix[x][y] > num:
                    return matrix[x][y]
            x += 1
            y += 1

    dim = 5
    matrix = [[0 for x in range(dim)] for x in range(dim)]
    assert traverse(matrix, 60) == 122
    matrix = [[0 for x in range(dim)] for x in range(dim)]
    assert traverse(matrix, 800) == 806

    num = 325489
    dim = int(math.sqrt(num))
    if dim % 2 == 0:
        dim += 1
    matrix = [[0 for x in range(dim)] for x in range(dim)]
    print("day3_part_two:", traverse(matrix, num))

#------------------------------------------------------------------------------#
def day4():

    def validate(passw):
        words = dict()
        for word in passw.split(' '):
            if word in words:
                return False
            else:
                words[word] = 1
        return True

    assert validate("aa bb cc dd ee") == True
    assert validate("aa bb cc dd aa") == False
    assert validate("aa bb cc dd aaa") == True

    valid = 0
    for passw in read_file('day4.txt'):
        if validate(passw) is True:
            valid += 1
    print("day4:", valid)


def day4_part_two():

    def validate(passw):
        words = dict()
        for word in passw.split(' '):
            word = "".join(sorted(word))
            if word in words:
                return False
            else:
                words[word] = 1
        return True

    assert validate("abcde fghij") == True
    assert validate("abcde xyz ecdab") == False
    assert validate("a ab abc abd abf abj") == True
    assert validate("iiii oiii ooii oooi oooo") == True
    assert validate("oiii ioii iioi iiio") == False

    valid = 0
    for passw in read_file('day4.txt'):
        if validate(passw) is True:
            valid += 1
    print("day4_part_two:", valid)


#------------------------------------------------------------------------------#
def day5():

    def jump(offsets):
        jumps = 0
        pos = 0
        len_ = len(offsets)
        while pos >= 0 and pos < len_:
            pos_ = pos + offsets[pos]
            offsets[pos] += 1
            pos = pos_
            jumps += 1
        return jumps

    assert jump([0, 3, 0, 1, -3]) == 5

    offsets = list()
    for offset in read_file('day5.txt'):
        offsets.append(int(offset))
    print("day5:", jump(offsets))


def day5_part_two():

    def jump(offsets):
        jumps = 0
        pos = 0
        len_ = len(offsets)
        while pos >= 0 and pos < len_:
            pos_ = pos + offsets[pos]
            if offsets[pos] >= 3:
                offsets[pos] -= 1
            else:
                offsets[pos] += 1
            pos = pos_
            jumps += 1
        return jumps

    assert jump([0, 3, 0, 1, -3]) == 10

    offsets = list()
    for offset in read_file('day5.txt'):
        offsets.append(int(offset))
    print("day5_two:", jump(offsets))


#------------------------------------------------------------------------------#
def day6():

    def update(m):
        len_ = len(m)
        max_ = max(m)
        i = [i for i, v in enumerate(m) if v == max_][0]
        m[i] = 0
        while max_ > 0:
            i = (i + 1) % len_
            m[i] += 1
            max_ -= 1
        return m

    def realloc(m):
        seen = dict()
        len_ = len(m)
        iterations = 0
        seen["".join(["%03d" % x for x in m])] = 1
        while True:
            iterations += 1
            m = update(m)
            pattern = "".join(["%03d" % x for x in m])
            if pattern in seen:
                return iterations
            seen[pattern] = iterations

    assert realloc([0, 2, 7, 0]) == 5

    mem = [int(x) for x in re.split(" |\t", read_file('day6.txt')[0])]
    print("day6:", realloc(mem))


def day6_part_two():
    def update(m):
        len_ = len(m)
        max_ = max(m)
        i = [i for i, v in enumerate(m) if v == max_][0]
        m[i] = 0
        while max_ > 0:
            i = (i + 1) % len_
            m[i] += 1
            max_ -= 1
        return m

    def realloc_diff(m):
        seen = dict()
        len_ = len(m)
        iterations = 0
        seen["".join(["%03d" % x for x in m])] = 1
        while True:
            iterations += 1
            m = update(m)
            pattern = "".join(["%03d" % x for x in m])
            if pattern in seen:
                return iterations - seen[pattern]
            seen[pattern] = iterations

    assert realloc_diff([0, 2, 7, 0]) == 4

    mem = [int(x) for x in re.split(" |\t", read_file('day6.txt')[0])]
    print("day6_part_two:", realloc_diff(mem))


#------------------------------------------------------------------------------#
def _day7_parse_input(input_):
    for line in input_:
        m = re.match(r'^(\S+)\s+\((\d+)\)(?:\s+->\s+(.+))?', line)
        node = m.group(1)
        weight = m.group(2)
        children = list()
        if m.group(3):
            children = m.group(3).split(', ')
        yield node, weight, children

def _day7_create_tree(input_):
    tree = dict()
    for node, weight, children in _day7_parse_input(input_):
        if not node in tree:
            tree[node] = dict()
        tree[node]['weight'] = int(weight)
        tree[node]['children'] = children
        for child in children:
            if not child in tree:
                tree[child] = dict()
            tree[child]['parent'] = node
    return tree


def day7():

    def find_root(input_):
        tree = _day7_create_tree(input_)
        for node in tree:
            if 'parent' not in tree[node]:
                return node

    test_ = """pbga (66)
xhth (57)
ebii (61)
havc (66)
ktlj (57)
fwft (72) -> ktlj, cntj, xhth
qoyq (66)
padx (45) -> pbga, havc, qoyq
tknk (41) -> ugml, padx, fwft
jptl (61)
ugml (68) -> gyxo, ebii, jptl
gyxo (61)
cntj (57)"""
    assert find_root(test_.split('\n')) == 'tknk'
    print("day7:", find_root(read_file('day7.txt')))

def day7_part_two():

    def find_root(tree):
        for node in tree:
            if 'parent' not in tree[node]:
                return node

    def weight_of_tree(node, tree):
        weight = tree[node]['weight']
        for child in tree[node]['children']:
            weight += weight_of_tree(child, tree)
        return weight

    def traverse(node, tree):
        subtree = dict()
        for child in tree[node]['children']:
            weight = weight_of_tree(child, tree)
            if weight not in subtree:
                subtree[weight] = list()
            subtree[weight].append(child)
        try:
            node_ = [subtree[k][0] for k in subtree.keys() if len(subtree[k]) == 1][0]
        except:
            # all subtrees are equal
            parent = tree[node]['parent']
            diff = weight_of_tree([child for child in tree[parent]['children'] if child != node][0], tree) -  weight_of_tree(node, tree)
            return (node, tree[node]['weight'], tree[node]['weight']+diff)
        return traverse(node_, tree)

    test_ = """pbga (66)
xhth (57)
ebii (61)
havc (66)
ktlj (57)
fwft (72) -> ktlj, cntj, xhth
qoyq (66)
padx (45) -> pbga, havc, qoyq
tknk (41) -> ugml, padx, fwft
jptl (61)
ugml (68) -> gyxo, ebii, jptl
gyxo (61)
cntj (57)"""
    tree = _day7_create_tree(test_.split('\n'))
    root = find_root(tree)
    assert weight_of_tree('ugml', tree) == 251
    assert weight_of_tree('padx', tree) == 243
    assert weight_of_tree('fwft', tree) == 243
    assert weight_of_tree('tknk', tree) == 778
    assert traverse(root, tree) == ('ugml', 68, 60)

    tree = _day7_create_tree(read_file('day7.txt'))
    root = find_root(tree)
    print("day7_part_two: ", traverse(root, tree))


#------------------------------------------------------------------------------#

def _day8_parse_input(input_):
    for line in input_:
        line = line.split(' ')
        reg = "__" + line[0]
        op = line[1]
        num = int(line[2])
        cond = "if __%s" % " ".join(line[4:])
        yield reg, op, num, cond

_day8_tr = {
    'inc': '+',
    'dec': '-',
}

def day8():

#     __test = """b inc 5 if a > 1
# a inc 1 if b < 5
# c dec -10 if a >= 1
# c inc -20 if c == 10"""

    regs = dict()
    code = list()
    # for reg, op, num, cond in _day8_parse_input(test.split('\n')):
    for reg, op, num, cond in _day8_parse_input(read_file('day8.txt')):
        regs[reg] = 0
        code.append("%s: %s %s= %d" % (cond, reg, _day8_tr[op], num))

    for reg in regs:
        code.insert(0, "%s = 0" % reg)

    exec("\n".join(code))
    vars = list()
    for reg in regs:
        vars.append(locals()[reg])
    # assert max(vars) == 1
    print("day8: ", max(vars))


def day8_part_two():

    regs = dict()
    code = list()
    # for reg, op, num, cond in _day8_parse_input(test.split('\n')):
    for reg, op, num, cond in _day8_parse_input(read_file('day8.txt')):
        regs[reg] = 0
        code.append("%s: %s %s= %d" % (cond, reg, _day8_tr[op], num))

    for reg in regs:
        code.insert(0, "%s = 0" % reg)

    for c in code:
        exec(c)
        for reg in regs:
            try:
                if regs[reg] < locals()[reg]:
                    regs[reg] = locals()[reg]
            except:
                pass

    print("day8_part_two: ", max([regs[r] for r in regs.keys()]))

#------------------------------------------------------------------------------#
def day9():

    def groups(s):
        s = list(s)
        i = 0
        current_group = 0
        in_garbage = False
        sum_ = 0
        while i<len(s):
            if s[i] == '!':
                i += 1
            elif s[i] == '{':
                if in_garbage == False:
                    current_group += 1
            elif s[i] == '<':
                in_garbage = True
            elif s[i] == '>':
                in_garbage = False
            elif s[i] == '}':
                if in_garbage == False:
                    sum_ += current_group
                    current_group -= 1
            i += 1
        return sum_

    assert groups('{}') == 1
    assert groups('{{{}}}') == 6
    assert groups('{{},{}}') == 5
    assert groups('{{{},{},{{}}}}') == 16
    assert groups('{<a>,<a>,<a>,<a>}') == 1
    assert groups('{{<ab>},{<ab>},{<ab>},{<ab>}}') == 9
    assert groups('{{<!!>},{<!!>},{<!!>},{<!!>}}') == 9
    assert groups('{{<a!>},{<a!>},{<a!>},{<ab>}}') == 3

    print("day9:", groups("".join(read_file('day9.txt'))))

def day9_part_two():
    def cleanup(s):
        s = list(s)
        i = 0
        in_garbage = False
        sum_ = 0
        while i<len(s):
            if s[i] == '!':
                i += 1
            elif s[i] == '>':
                in_garbage = False
            elif in_garbage:
                sum_ += 1
            elif s[i] == '<':
                in_garbage = True
            i += 1
        return sum_

    assert cleanup('<>') == 0
    assert cleanup('<random characters>') == 17
    assert cleanup('<<<<>') == 3
    assert cleanup('<{!>}>') == 2
    assert cleanup('<!!>') == 0
    assert cleanup('<!!!>>') == 0
    assert cleanup('<{o"i!a,<{i<a>') == 10

    print("day9_part_two:", cleanup("".join(read_file('day9.txt'))))


#------------------------------------------------------------------------------#
if __name__ == '__main__':
    day1()
    day1_part_two()
    day2()
    day2_part_two()
    day3()
    day3_part_two()
    day4()
    day4_part_two()
    day5()
    day5_part_two()
    day6()
    day6_part_two()
    day7()
    day7_part_two()
    day8()
    day8_part_two()
    day9()
    day9_part_two()