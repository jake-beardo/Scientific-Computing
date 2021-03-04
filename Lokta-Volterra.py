


def dx_dt(x,y,a,d):
  dx_dt = x(1-x) - (a*x*y)/(d+x)
  return dx_dt
