def summation_i_squared(n):
  if not isinstanse(n ,int) or n < 0:
    return none
  
  total_sum = (n * (n + 1) * (2 * n + 1)) // 6
    
  return total_sum
