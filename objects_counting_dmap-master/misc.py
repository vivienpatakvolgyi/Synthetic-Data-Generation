
def avg(array):
    sum = 0
    for x in array:
        sum += x
    if len(array) == 0:
        return 0
    else:
        return sum / len(array)


def abs(num):
    if num < 0:
        return num * -1
    else:
        return num
