__author__ = 'dot'


def show(mat):
    sb = []
    for x in range(len(mat)):
        sb1 = []
        for u in range(len(mat[x])):
            sb1.append("%.3f" % mat[x][u])
        line = ",".join(sb1)
        sb.append(("%3d: " % (x)) + line)
    return "\n".join(sb)


def show_vec(vec,separator):
    sb = []
    for x in range(len(vec)):
        sb.append(str(vec[x]))
    return separator.join(sb)
