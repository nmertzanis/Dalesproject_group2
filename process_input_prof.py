import numpy as np

# Profile for day_min
profile = open("prof.inp.001.txt", "w")
profile.write("#ASTEX case using prescribed vertical grid, Nlev = 427 1\n \
      height(m)   thl(K)     qt(kg/kg)       u(m/s)     v(m/s)     tke(m2/s2)\n")


def format_num(n):
    return '{:.7s}'.format('{:0.4f}'.format(n))

def add_line(height, thl, qt, u, v, tke):

    return "      " + format_num(height) + "      " \
    + format_num(thl) + "      " + format_num(qt) + "      " \
    + format_num(u) + "      " + format_num(v) + "      " \
    + format_num(tke) + "\n"

profile = open("input/prof.inp.001.txt", "a")

#profile.write(add_line(10.4545654,1,1,1,1,1))

def make_heights(height_base, step, iterations):

    heights = []
    heights.append(height_base)
    for i in range(1, iterations):
        heights.append(heights[i-1] + step)

    return heights

heights = make_heights(0, 101)

def make_rest(heights, delta_th, qta, qtb, ua, ub, va, vb, tkea, tkeb):

    thls = []
    qts = []
    u = []
    v = []
    tke = []

    for i in range(0, len(heights)):

        if heights[i] < 300:
            thls.append(295)
            qts.append(qta)
            u.append(ua)
            v.append(va)
            tke.append(tkea)
        else:
            u.append(ub)
            v.append(vb)
            qts.append(qtb)
            tke.append(tkeb)
            thls.append(295 + delta_th + 0.005 * heights[i])

    return thls, qts, u, v, tke

thls, qts, u, v, tke = make_rest(heights, delta_th=10, qta=0, qtb=0, ua=1, ub=1, va=0, vb=0, tkea=1, tkeb=0)

for i in range(0, 80):
    profile.write(add_line(heights[i], thls[i], qts[i], u[i], v[i], tke[i]))



profile.close()
