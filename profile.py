import cProfile

def my_func():
    return sum([i for i in range(100)])

cProfile.run('my_func()')